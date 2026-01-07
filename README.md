# Adversarial Training Improves Transferability

## Project Overview

This project aims to reproduce the core experiments from the NIPS paper "Do Adversarially Robust ImageNet Models Transfer Better?", exploring the impact of Adversarial Training on model transferability. By performing adversarial training on the ImageNet dataset and evaluating the trained models' transfer performance on downstream tasks (such as classification), it verifies whether adversarial training can enhance model generalization and robustness.

The project consists of two main components:
- **Adversarial Training Module**: Performs distributed adversarial training on ImageNet using PGD attacks to generate robust pre-trained models.
- **Transfer Evaluation Module**: Transfers pre-trained models to downstream datasets (e.g., CIFAR-10, CIFAR-100), evaluating performance in fixed-feature and full-network fine-tuning modes.

The experimental results will help understand how adversarial training affects model transfer learning capabilities, particularly whether it leads to better performance on downstream tasks.

## Main Experiments

- **ImageNet Adversarial Training**: Trains ResNet series models on ImageNet using Distributed Data Parallel (DDP), employing PGD or L2 norm adversarial sample generation.
- **Downstream Transfer Evaluation**: Tests the transfer performance of pre-trained models on multiple downstream datasets, supporting linear classification and full-network fine-tuning modes, recording accuracy metrics.

## Experimental Details

### ImageNet Pre-training
- **Models**: ResNet-18, ResNet-50
- **Training Setup**: 90 epochs, SGD optimizer with initial learning rate 0.1, momentum 0.9, weight decay 1e-4, learning rate decay at epochs 30 and 60 with gamma 0.1
- **Adversarial Training**: PGD attacks with 3 steps, alpha = 2 * $\epsilon$ / 3, $\epsilon$ in {0.0 (ERM), 0.01 to 5.0 (L2), 0.5/255 to 8.0/255 (L∞)}
- **Data Augmentation**: Random resized crop (224x224), random horizontal flip, ImageNet normalization
- **Distributed Training**: Multi-GPU with torchrun and DDP (8x NVIDIA GeForce RTX 3090 GPUs)

### Downstream Transfer Evaluation
- **Datasets**: CIFAR-10, CIFAR-100
- **Training Setup**: 150 epochs, SGD optimizer with learning rates [0.01, 0.001], momentum 0.9, weight decay 5e-4, learning rate decay at epochs 50 and 100 with gamma 0.1
- **Modes**: Fixed-feature (linear classifier on frozen backbone) and full-network fine-tuning
- **Metrics**: Per-sample and per-class accuracy
- **Data Augmentation**: Random resized crop (224x224), random horizontal flip for training; resize (256x256) + center crop (224x224) for testing
- **Validation**: 10% split from training set for hyperparameter tuning

## Results — Transfer Accuracy (tables)

Below are the reproduced transfer-accuracy tables (CIFAR-10 / CIFAR-100) for different robustness parameters ($\epsilon$). Values are percentages; bolded entries indicate the best value in the row.

#### Transfer accuracy for L∞ adversarial training:

| Dataset | Transfer Type | Model | 0.0 | 0.5/255 | 1.0/255 | 2.0/255 | 4.0/255 | 8.0/255 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | Full-network | ResNet-18 | 95.81 | 96.35 | 96.68 | 96.73 | **96.84** | 96.47 |
| CIFAR-10 | Full-network | ResNet-50 | 96.98 | 97.54 | 97.76 | 97.61 | **97.76** | 97.59 |
| CIFAR-10 | Fixed-feature | ResNet-18 | 73.72 | 85.98 | 88.42 | 89.36 | **89.42** | 88.43 |
| CIFAR-10 | Fixed-feature | ResNet-50 | 76.51 | 88.22 | 90.38 | 91.73 | **92.55** | 91.93 |
| CIFAR-100 | Full-network | ResNet-18 | 80.48 | 82.32 | 83.06 | **83.35** | 82.59 | 81.45 |
| CIFAR-100 | Full-network | ResNet-50 | 83.95 | 85.09 | 85.25 | 85.69 | **85.74** | 84.96 |
| CIFAR-100 | Fixed-feature | ResNet-18 | 51.52 | 65.67 | 68.24 | **70.23** | 69.60 | 67.78 |
| CIFAR-100 | Fixed-feature | ResNet-50 | 54.22 | 68.76 | 72.97 | 73.97 | **74.62** | 72.94 |

#### Transfer accuracy for L2 adversarial training:

| Dataset | Transfer Type | Model | 0 | 0.01 | 0.03 | 0.05 | 0.1 | 0.25 | 0.5 | 1.0 | 3.0 | 5.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | Full-network | ResNet-18 | 95.81 | 95.71 | 96.14 | 96.17 | 96.24 | 96.52 | 96.41 | 96.68 | **96.77** | 96.41 |
| CIFAR-10 | Full-network | ResNet-50 | 96.98 | 97.18 | 97.23 | 96.90 | 97.48 | 97.62 | 97.72 | **97.78** | 97.76 | 97.64 |
| CIFAR-10 | Fixed-feature | ResNet-18 | 73.72 | 76.12 | 76.84 | 79.17 | 81.63 | 84.66 | 86.37 | 89.36 | **90.97** | 89.83 |
| CIFAR-10 | Fixed-feature | ResNet-50 | 76.51 | 78.48 | 82.05 | 82.02 | 86.13 | 87.97 | 89.68 | 91.36 | **93.52** | 93.50 |
| CIFAR-100 | Full-network | ResNet-18 | 80.48 | 81.09 | 81.23 | 81.28 | 81.39 | 82.63 | 82.71 | 82.86 | **82.93** | 81.68 |
| CIFAR-100 | Full-network | ResNet-50 | 83.95 | 83.90 | 84.41 | 84.01 | 84.61 | 85.29 | 85.51 | 85.54 | **85.79** | 85.60 |
| CIFAR-100 | Fixed-feature | ResNet-18 | 51.52 | 54.38 | 54.83 | 57.79 | 60.62 | 64.61 | 67.48 | 70.91 | **71.61** | 69.70 |
| CIFAR-100 | Fixed-feature | ResNet-50 | 54.22 | 55.49 | 60.52 | 59.84 | 64.22 | 69.33 | 71.54 | 74.32 | **77.16** | 75.62 |

Notes:
- Results are reported as top-1 transfer accuracy (%) on each downstream dataset and transfer mode.
- The first table shows perturbation magnitudes for L∞ adversarial training (scaled to the image range). The second table reports a broader set of robustness parameters for L2 adversarial training.

## Key Findings

Adversarial training significantly improves model transfer performance on downstream tasks, for both L2 and L∞ norms. Compared to L∞, L2 adversarial training achieves higher maximum transfer performance on downstream tasks, particularly in fixed-feature transfer mode.

## Dependencies

- Python 3.8+
- PyTorch 1.10+
- TorchVision
- TorchAttacks
- Other standard libraries (e.g., NumPy, PIL)

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Pony-Li/Adversarial-Training-Improves-Transferability.git
   cd Adversarial-Training-Improves-Transferability
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare data: Download ImageNet and downstream datasets, and set the corresponding paths.

4. Run experiments: Configure parameters as needed to start adversarial training or transfer evaluation.

   Example commands:
   ```bash
   # Adversarial training (using 4 GPUs)
   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 ./scripts/run_experiment.sh train --data-root /path/to/imagenet --arch resnet50 --norm l2 --epsilon 3.0 --batch-size 512 --num-workers 8 --save-dir ./ckpts/

   # Transfer evaluation (using 4 GPUs)
   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 ./scripts/run_experiment.sh finetune --data-root /path/to/datasets --dataset CIFAR10 --arch resnet50 --ckpt /path/to/ckpt.pth --mode fixed --batch-size 64 --num-workers 4 --save-dir ./transfer_ckpts
   ```

## Citation

If you use this project in your research, please cite the original paper:

```
@article{salman2020adversarially,
  title={Do adversarially robust imagenet models transfer better?},
  author={Salman, Hadi and Ilyas, Andrew and Engstrom, Logan and Kapoor, Ashish and Madry, Aleksander},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={3533--3545},
  year={2020}
}
```
