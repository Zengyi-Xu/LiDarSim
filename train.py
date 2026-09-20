"""训练脚本"""
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import *
from models.pointnet import PointNetClassifier, pointnet_loss
from utils.dataset import ModelNet40

# 固定随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for points, labels in tqdm(loader, desc="Training"):
        points, labels = points.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, trans = model(points)
        loss = pointnet_loss(outputs, labels, trans)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * points.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for points, labels in tqdm(loader, desc="Evaluating"):
            points, labels = points.to(device), labels.to(device)
            outputs, trans = model(points)
            loss = pointnet_loss(outputs, labels, trans)

            total_loss += loss.item() * points.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    # 数据集
    train_set = ModelNet40(DATA_DIR, split='train', num_points=NUM_POINTS)
    test_set = ModelNet40(DATA_DIR, split='test', num_points=NUM_POINTS)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    print(f"Train: {len(train_set)}, Test: {len(test_set)}")

    # 模型
    model = PointNetClassifier(num_classes=NUM_CLASSES,
                               feature_transform=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = eval_epoch(model, test_loader, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        epoch_time = time.time() - start
        print(f"Epoch [{epoch}/{EPOCHS}] {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

        # 保存最优模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / 'best_model.pth')
            print(f"  -> New best accuracy: {best_acc:.4f}")

        # 定期保存
        if epoch % SAVE_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history,
            }, CHECKPOINT_DIR / f'checkpoint_epoch{epoch}.pth')

    print(f"\nTraining complete. Best test accuracy: {best_acc:.4f}")
    np.save(OUTPUT_DIR / 'history.npy', history)


if __name__ == '__main__':
    main()
