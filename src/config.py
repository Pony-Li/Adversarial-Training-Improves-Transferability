# config.py
"""
Configuration constants for adversarial training and transfer evaluation.
"""

# Adversarial Training Constants (from AT.py)
NUM_EPOCHS = 90
INIT_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
MILESTONES = [30, 60]
GAMMA = 0.1
PGD_STEPS = 3  # 训练时 PGD 迭代次数

# Transfer Evaluation Constants (from finetune_eval.py)
FIXED_LRS = [0.01, 0.001]        # 初始学习率候选
FIXED_EPOCHS = 150               # 总 epoch 数
LR_MILESTONES = [50, 100]        # 每 50 epoch ×0.1
VAL_SPLIT = 0.1                  # 无官方 val 集时，从 train 随机划出 10% 做 val