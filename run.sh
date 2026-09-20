#!/bin/bash
set -e

echo "===== LiDAR-PointNet 一键部署 ====="

# 1. 环境检查
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python >= 3.9"
    exit 1
fi

# 2. 安装依赖(若未安装)
echo "[1/4] Checking dependencies..."
pip install -q torch torchvision numpy h5py tqdm matplotlib 2>/dev/null || true

# 3. 下载数据
echo "[2/4] Downloading ModelNet40..."
python download_data.py

# 4. 训练
echo "[3/4] Starting training..."
python train.py

# 5. 评估
echo "[4/4] Evaluating best model..."
python eval.py

echo "===== All done! Check outputs/ and checkpoints/ ====="
