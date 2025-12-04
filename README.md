# AT-Transfer-ImageNet

Reproduction project for the ImageNet adversarial training part of  
**“Do Adversarially Robust ImageNet Models Transfer Better?”**

This repo focuses on training **standard** and **adversarially robust** ImageNet models (L2 / L∞ PGD) with **DDP**, and preparing checkpoints for later transfer-learning experiments.

---

## 1. Features

- ImageNet training with:
  - Standard ERM (no adversarial perturbation)
  - L2-PGD adversarial training
  - L∞-PGD adversarial training
- Distributed training with **torchrun + DDP**
- Backbones (easily extensible):
  - `resnet18`
  - `resnet34`
  - `resnet50`
  - `wide_resnet50_2`
- Adversarial example generation via **[torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch)**:
  - `PGDL2` for L2 norm
  - `PGD` for L∞ norm
- ImageNet-standard preprocessing (224×224, mean/std normalization)
- Automatic checkpoint management:
  - Save a snapshot once every **10 个 epoch**

---

## 2. Environment

Recommended environment (adjustable according to actual situation):

- Python ≥ 3.8
- PyTorch ≥ 1.12
- torchvision ≥ 0.13
- CUDA 11.x / 12.x
- torchattacks ≥ 3.4

Install dependencies:

```bash
pip install torch torchvision
pip install torchattacks
