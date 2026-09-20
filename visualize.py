"""点云分类结果可视化

生成两类图到 outputs/visualizations/:
1. result_grid.png   - 测试集样本的点云 3D 散点图, 标注预测 vs 真实类别(绿色正确/红色错误)
2. tnet_alignment.png - 单个样本的 Input T-Net 对齐效果(原始点云 vs 对齐后点云)
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from config import DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR, NUM_POINTS, NUM_CLASSES, DEVICE
from models.pointnet import PointNetClassifier
from utils.dataset import ModelNet40

# ModelNet40 标准类别名 (按 HDF5 label 索引顺序)
CLASS_NAMES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox',
]

VIZ_DIR = OUTPUT_DIR / 'visualizations'


def plot_pointcloud(ax, points, title, color=None, size=3):
    ax.scatter(points[:, 0], points[:, 2], points[:, 1],
               c=color, s=size, cmap='viridis', alpha=0.7, edgecolors='none')
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    ax.view_init(elev=30, azim=45)
    ax.set_box_aspect((1, 1, 1))


@torch.no_grad()
def visualize_results(n_show=8):
    """随机选取测试样本, 展示预测结果"""
    device = torch.device(DEVICE)
    test_set = ModelNet40(DATA_DIR, split='test', num_points=NUM_POINTS)

    model = PointNetClassifier(num_classes=NUM_CLASSES, feature_transform=True).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / 'best_model.pth',
                                     map_location=device, weights_only=True))
    model.eval()

    indices = np.random.RandomState(0).choice(len(test_set), n_show * 3, replace=False)
    samples = []
    for idx in indices:
        pts, label = test_set[idx]
        out, _ = model(pts.unsqueeze(0).to(device))
        pred = out.argmax(1).item()
        samples.append((pts.numpy(), label.item(), pred))
        if len(samples) >= n_show:
            break

    correct = sum(1 for _, t, p in samples if t == p)
    fig = plt.figure(figsize=(16, 4))
    for i, (pts, label, pred) in enumerate(samples):
        ax = fig.add_subplot(2, n_show // 2, i + 1, projection='3d')
        color = '#2ca02c' if label == pred else '#d62728'
        mark = '✓' if label == pred else '✗'
        title = f"{mark} pred: {CLASS_NAMES[pred]}\ntrue: {CLASS_NAMES[label]}"
        plot_pointcloud(ax, pts, title, color=color)
    fig.suptitle(f"PointNet on ModelNet40 (test samples, {correct}/{len(samples)} correct)",
                 fontsize=12)
    fig.tight_layout()

    out_path = VIZ_DIR / 'result_grid.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return samples


@torch.no_grad()
def visualize_tnet_alignment():
    """展示 Input T-Net 的点云对齐效果"""
    device = torch.device(DEVICE)
    test_set = ModelNet40(DATA_DIR, split='test', num_points=NUM_POINTS)

    model = PointNetClassifier(num_classes=NUM_CLASSES, feature_transform=True).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / 'best_model.pth',
                                     map_location=device, weights_only=True))
    model.eval()

    pts, label = test_set[0]
    x = pts.unsqueeze(0).to(device)                    # (1, N, 3)

    # 复现 encoder 内部的 input transform
    trans = model.feat.input_transform(x.transpose(2, 1))  # (1, 3, 3)
    aligned = torch.bmm(x, trans).squeeze(0).cpu().numpy()
    original = pts.numpy()
    t = trans.squeeze(0).cpu().numpy()

    depth_o = original[:, 2]
    depth_a = aligned[:, 2]

    fig = plt.figure(figsize=(10, 4.5))
    ax1 = fig.add_subplot(121, projection='3d')
    sc = ax1.scatter(original[:, 0], original[:, 2], original[:, 1],
                     c=depth_o, s=3, cmap='viridis', edgecolors='none')
    ax1.set_title(f'Original: {CLASS_NAMES[label.item()]}', fontsize=10)
    ax1.set_axis_off(); ax1.view_init(elev=30, azim=45); ax1.set_box_aspect((1, 1, 1))

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(aligned[:, 0], aligned[:, 2], aligned[:, 1],
                c=depth_a, s=3, cmap='viridis', edgecolors='none')
    ax2.set_title('After Input T-Net', fontsize=10)
    ax2.set_axis_off(); ax2.view_init(elev=30, azim=45); ax2.set_box_aspect((1, 1, 1))

    det = float(np.linalg.det(t))
    orth_err = float(np.linalg.norm(t @ t.T - np.eye(3)))
    fig.suptitle(f"T-Net alignment | det(R)={det:.3f}, orthogonality error={orth_err:.4f}",
                 fontsize=11)
    fig.tight_layout()

    out_path = VIZ_DIR / 'tnet_alignment.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"T-Net matrix:\n{np.round(t, 3)}")


if __name__ == '__main__':
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    visualize_results()
    visualize_tnet_alignment()
