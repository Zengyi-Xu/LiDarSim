"""ModelNet40 HDF5 数据集加载"""
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ModelNet40(Dataset):
    def __init__(self, root, split='train', num_points=1024):
        self.root = root
        self.split = split
        self.num_points = num_points

        # 自动定位 HDF5 文件
        if split == 'train':
            files = [f for f in os.listdir(root)
                     if 'train' in f and f.endswith('.h5')]
        else:
            files = [f for f in os.listdir(root)
                     if 'test' in f and f.endswith('.h5')]
        files.sort()

        self.points = []
        self.labels = []
        for f in files:
            with h5py.File(os.path.join(root, f), 'r') as hf:
                self.points.append(hf['data'][:])
                self.labels.append(hf['label'][:].flatten())

        self.points = np.concatenate(self.points, axis=0)   # (N, 2048, 3)
        self.labels = np.concatenate(self.labels, axis=0)   # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pts = self.points[idx][:self.num_points]  # (num_points, 3)

        # 随机采样(训练时增强)
        if self.split == 'train' and pts.shape[0] > self.num_points:
            choice = np.random.choice(pts.shape[0], self.num_points, replace=False)
            pts = pts[choice, :]

        # 数据增强: 随机旋转 + 抖动
        if self.split == 'train':
            pts = self._augment(pts)

        pts = torch.from_numpy(pts).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return pts, label

    def _augment(self, pts):
        # 随机绕 Y 轴旋转
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, 0, sin_a],
                      [0, 1, 0],
                      [-sin_a, 0, cos_a]])
        pts = pts @ R.T

        # 随机抖动
        jitter = np.random.normal(0, 0.02, pts.shape)
        pts = pts + jitter

        return pts.astype(np.float32)
