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

4. Run experiments: Configure parameters as needed to start adversarial training or transfer evaluation (specific commands will be updated after code refactoring).

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
