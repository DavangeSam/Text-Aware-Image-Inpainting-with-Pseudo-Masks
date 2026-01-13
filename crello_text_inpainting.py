
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from datasets import load_dataset
import numpy as np, cv2, matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import lpips

# Load dataset
ds = load_dataset("cyberagent/crello", split="train[:2000]")
print(ds)

# Sample visualization
sample = ds[0]["image"]
plt.imshow(sample)
plt.title("Sample Crello image")
plt.axis("off")
plt.show()

#  Text detection 
def detect_text_mask(img):
    H, W = img.shape[:2]
    newW, newH = (320, 320)
    rW, rH = W/newW, H/newH
    blob = cv2.dnn.blobFromImage(img, 1.0, (newW,newH),
                                 (123.68,116.78,103.94), True, False)
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    net.setInput(blob)
    scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid",
                                    "feature_fusion/concat_3"])
    confThreshold = 0.5
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    for y in range(numRows):
        scoresData = scores[0,0,y]
        xData0, xData1, xData2, xData3, anglesData =             geometry[0,0,y], geometry[0,1,y], geometry[0,2,y], geometry[0,3,y], geometry[0,4,y]
        for x in range(numCols):
            if scoresData[x] < confThreshold: continue
            angle = anglesData[x]; cos, sin = np.cos(angle), np.sin(angle)
            h = xData0[x] + xData2[x]; w = xData1[x] + xData3[x]
            endX = int(x * 4.0 + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(y * 4.0 - (sin * xData1[x]) + (cos * xData2[x]))
            startX, startY = int(endX - w), int(endY - h)
            rects.append((int(startX*rW), int(startY*rH), int(endX*rW), int(endY*rH)))
    mask = np.zeros((H, W), np.uint8)
    for (x1,y1,x2,y2) in rects:
        cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
    return mask

#  Dataset wrapper
class CrelloDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.tf = transform
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        img = np.array(self.ds[idx]["image"].convert("RGB"))
        mask = detect_text_mask(img)
        img_t = self.tf(img)
        mask_t = torch.tensor(mask/255.0).unsqueeze(0).float()
        return img_t, mask_t

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

trainset = CrelloDataset(ds.select(range(0,1500)), transform)
valset   = CrelloDataset(ds.select(range(1500,1800)), transform)
trainloader = DataLoader(trainset, batch_size=8, shuffle=True)

#  U-Net segmentation
class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights='IMAGENET1K_V1')
        self.encoder = nn.Sequential(*list(base.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512,256,2,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256,128,2,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,2,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,2,stride=2),
            nn.ReLU(),
            nn.Conv2d(32,1,1)
        )
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x

model = UNetSmall().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
bce = nn.BCELoss()

# Quick training loop
for epoch in range(1):
    model.train()
    total_loss = 0
    for img, mask in trainloader:
        img, mask = img.cuda(), mask.cuda()
        pred = model(img)
        loss = bce(pred, mask)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={total_loss/len(trainloader):.4f}")

#  Simple LaMa-style Inpainting model
class SimpleInpaintNet(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        self.enc1 = nn.Conv2d(in_ch, 64, 5, padding=2)
        self.enc2 = nn.Conv2d(64,128,3,stride=2,padding=1)
        self.enc3 = nn.Conv2d(128,256,3,stride=2,padding=1)
        self.dec1 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.dec2 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.outc = nn.Conv2d(64,3,3,padding=1)
    def forward(self, img, mask):
        x = torch.cat([img, mask], 1)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.outc(x))
        return x

inpaint = SimpleInpaintNet().cuda()
opt = torch.optim.Adam(inpaint.parameters(), lr=1e-4)
L1 = nn.L1Loss()

for step, (img, mask) in enumerate(trainloader):
    img, mask = img.cuda(), mask.cuda()
    corrupted = img * (1-mask)
    pred = inpaint(corrupted, mask)
    loss = L1(pred*(1-mask), img*(1-mask))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(f"Step {step} Loss {loss.item():.4f}")
    if step > 200: break

#  Evaluation
lpips_model = lpips.LPIPS(net='alex')

def evaluate(img, mask):
    img = img.unsqueeze(0).cuda()
    mask = mask.unsqueeze(0).cuda()
    with torch.no_grad():
        corrupted = img*(1-mask)
        pred = inpaint(corrupted, mask)
    recon = pred.cpu()[0]
    orig  = img.cpu()[0]
    mse = F.mse_loss(recon, orig).item()
    psnr = 10*np.log10(1/mse)
    ssim_val = ssim(orig.permute(1,2,0).numpy(), recon.permute(1,2,0).numpy(),
                    channel_axis=2, data_range=1.0)
    lp = lpips_model(orig.unsqueeze(0), recon.unsqueeze(0)).item()
    return psnr, ssim_val, lp, recon

img, mask = valset[20]
psnr, ssim_val, lp, recon = evaluate(img, mask)
print(f"PSNR={psnr:.2f}, SSIM={ssim_val:.3f}, LPIPS={lp:.3f}")

plt.figure(figsize=(12,4))
plt.subplot(1,4,1); plt.imshow(img.permute(1,2,0)); plt.title("Original")
plt.subplot(1,4,2); plt.imshow(mask[0], cmap='gray'); plt.title("Mask")
plt.subplot(1,4,3); plt.imshow((img*(1-mask)).permute(1,2,0)); plt.title("Corrupted")
plt.subplot(1,4,4); plt.imshow(recon.permute(1,2,0)); plt.title("Inpainted")
plt.show()
