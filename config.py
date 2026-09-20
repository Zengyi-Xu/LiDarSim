"""全局配置"""
import os
from pathlib import Path

import torch

# 路径
ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data" / "modelnet40_ply_hdf5_2048"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
OUTPUT_DIR = ROOT_DIR / "outputs"

for d in [CHECKPOINT_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 数据参数
NUM_POINTS = 1024          # 每样本采样点数 (PointNet 原文用 1024)
NUM_CLASSES = 40           # ModelNet40

# 训练参数
BATCH_SIZE = 32            # GPU 可调 64
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 其他
SEED = 42
SAVE_EVERY = 20            # 每 N epoch 保存检查点
