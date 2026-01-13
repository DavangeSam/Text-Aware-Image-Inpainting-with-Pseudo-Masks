
# Text Aware Image Inpainting with Pseudo Masks

This repository demonstrates a **lightweight, end-to-end pipeline for text removal and image inpainting** using weak supervision. Text regions are detected using a pre-trained text detector to generate *pseudo masks*, which are then used to train:

1. A **text segmentation model** (U-Net with ResNet18 encoder)
2. A **simple LaMa-style inpainting network**

The pipeline is trained and evaluated on the **Crello design dataset** from Hugging Face.

---

## ✨ Features

- No manual annotations required 
- Text segmentation using a ResNet-based U-Net
- Lightweight inpainting network inspired by LaMa
- Evaluation with PSNR, SSIM, and LPIPS
- End-to-end training and visualization in PyTorch



