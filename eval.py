"""评估脚本"""
import torch
import numpy as np
from torch.utils.data import DataLoader

from config import *
from models.pointnet import PointNetClassifier
from utils.dataset import ModelNet40


def main():
    device = torch.device(DEVICE)

    test_set = ModelNet40(DATA_DIR, split='test', num_points=NUM_POINTS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = PointNetClassifier(num_classes=NUM_CLASSES,
                               feature_transform=True).to(device)

    # 加载最优权重
    best_ckpt = CHECKPOINT_DIR / 'best_model.pth'
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_ckpt}")
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))

    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for points, labels in test_loader:
            points, labels = points.to(device), labels.to(device)
            outputs, _ = model(points)
            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f} ({correct}/{total})")

    np.savez(OUTPUT_DIR / 'predictions.npz',
             preds=np.array(all_preds), labels=np.array(all_labels))


if __name__ == '__main__':
    main()
