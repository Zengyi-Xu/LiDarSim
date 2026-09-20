# -*- coding: utf-8 -*-
"""点云分类 ANN 上限基准 (无模型规模限制) —— 供 GPU 机器运行.

回答: "3D 点云数据集不限模型规模能到多高的分类准确率?"
为博士后计划提供 ANN 天花板参照 (对比 SNN 扫描链 0.68 / 单次 HRRP 0.4).

数据: 父项目 data/modelnet40_{train,test}.parquet (2048 点/形状, 标签 0-39).
默认用前 10 类子集 (与 snn 实验一致), --classes 40 可切全量 (文献可比).

模型 (纯 torch, 不依赖 PyG/torch-points3d):
  pointnet      — 共享 MLP + 全局最大池化 (基线)
  pointnet2_ssg — Set Abstraction (fps + kNN 分组, 单尺度)
  dgcnn         — 动态 EdgeConv (kNN 图, 3 层)

训练: AdamW + cosine, label smoothing 0.1, 随机重采样/抖动/点丢弃增强.
评估: 每 epoch 固定重采样测一次; 训练结束做 T 次重采样投票 (TTA).

CPU 冒烟:  python pointcloud_ann_bench.py --smoke
GPU 参考:  python pointcloud_ann_bench.py --model dgcnn --n_points 1024 --epochs 200
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = Path(__file__).parent
DATA = HERE / ".." / "data"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- 数据 ----------------
class CloudSet(torch.utils.data.Dataset):
    def __init__(self, parquet, classes, n_points, train=False, seed=0):
        df = pd.read_parquet(parquet)
        df = df[df["label"].isin(classes)].reset_index(drop=True)
        self.cls_map = {c: i for i, c in enumerate(classes)}
        self.raw = [np.stack([np.asarray(q, dtype=np.float32) for q in p])
                    for p in df["inputs"].values]
        self.y = np.array([self.cls_map[v] for v in df["label"].values], dtype=np.int64)
        self.n_points, self.train = n_points, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.raw)

    def _one(self, i, rng):
        p = self.raw[i]
        if len(p) > self.n_points:
            sel = rng.choice(len(p), self.n_points, replace=False)
        elif len(p) < self.n_points:
            sel = rng.choice(len(p), self.n_points, replace=True)
        else:
            sel = np.arange(len(p))
        x = p[sel].copy()
        if self.train:  # 点丢弃 (to 85-100%) 再补齐
            keep = rng.uniform(0.85, 1.0)
            sub = rng.choice(len(x), max(1, int(len(x) * keep)), replace=False)
            x = x[sub]
            if len(x) < self.n_points:
                pad = x[rng.choice(len(x), self.n_points - len(x), replace=True)]
                x = np.concatenate([x, pad], axis=0)
        x -= x.mean(axis=0, keepdims=True)
        x /= (np.abs(x).max() + 1e-8)
        if self.train:
            x = x * rng.uniform(0.9, 1.1) + rng.normal(0, 0.005, x.shape).astype(np.float32)
        return torch.from_numpy(x), int(self.y[i])

    def __getitem__(self, i):
        return self._one(i, self.rng if self.train else np.random.default_rng(i))


def load_classes(k):
    df = pd.read_parquet(DATA / "modelnet40_train.parquet")
    return np.sort(df["label"].unique())[:k].tolist()


# ---------------- 基础算子 ----------------
def fps(x, n_out):
    """迭代最远点采样 x:(N,3)->(n_out,) 纯 torch"""
    n = x.shape[0]
    out = torch.zeros(n_out, dtype=torch.long)
    d = torch.full((n,), float("inf"))
    f = int(torch.randint(n, (1,)).item())
    for i in range(n_out):
        out[i] = f
        d = torch.minimum(d, ((x - x[f]) ** 2).sum(-1))
        f = int(d.argmax().item())
    return out


def knn_idx(x, k):
    """x:(B,N,C)->(B,N,k) 邻居索引 (不含自身)"""
    d = torch.cdist(x, x)
    return d.topk(k + 1, largest=False).indices[:, :, 1:]


def gather(x, idx):
    """x:(B,N,C), idx:(B,N,k)->(B,N,k,C)"""
    return x[torch.arange(x.shape[0])[:, None, None], idx]


# ---------------- 模型 ----------------
class PointNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, k))

    def forward(self, x):  # (B,N,3)
        h = self.mlp(x.transpose(1, 2))
        h = h.max(-1).values
        return self.head(h)


class SA(nn.Module):
    """Set Abstraction: fps 选中心 + kNN 分组 + 局部 mini-PointNet"""

    def __init__(self, n_out, k, cin, couts, global_ab=False):
        super().__init__()
        self.n_out, self.k, self.global_ab = n_out, k, global_ab
        layers, c = [], cin
        for co in couts:
            layers += [nn.Conv2d(c, co, 1), nn.BatchNorm2d(co), nn.ReLU()]
            c = co
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):  # (B,N,cin)
        B, N, C = x.shape
        if self.global_ab:
            g = x.unsqueeze(1).expand(B, 1, N, C)  # (B,1,N,C)
        else:
            ci = torch.stack([fps(x[b], self.n_out) for b in range(B)])
            centers = x[torch.arange(B)[:, None], ci]          # (B,S,C)
            idx = knn_idx(x, self.k)                            # (B,N,k)
            sidx = idx[torch.arange(B)[:, None], ci]            # (B,S,k)
            g = gather(x, sidx) - centers[:, :, None, :]        # (B,S,k,C)
        h = self.mlp(g.permute(0, 3, 1, 2)).max(-1).values      # (B,cout,S)
        return h.transpose(1, 2).contiguous() if not self.global_ab else h.squeeze(-1).unsqueeze(1)


class PointNet2SSG(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.sa1 = SA(512, 32, 3, [32, 32, 64])
        self.sa2 = SA(128, 32, 64, [64, 64, 128])
        self.sa3 = SA(None, None, 128, [128, 256, 512], global_ab=True)
        self.head = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, k))

    def forward(self, x):
        h = self.sa3(self.sa2(self.sa1(x)))
        return self.head(h[:, 0, :])


class EdgeConv(nn.Module):
    def __init__(self, cin, cout, k=32):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * cin, cout, 1), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 1), nn.BatchNorm2d(cout), nn.ReLU())

    def forward(self, x):  # (B,N,C)
        idx = knn_idx(x, self.k)
        nb = gather(x, idx)                          # (B,N,k,C)
        e = torch.cat([x[:, :, None, :].expand_as(nb), nb - x[:, :, None, :]], -1)
        return self.mlp(e.permute(0, 3, 1, 2)).max(-1).values.transpose(1, 2)


class DGCNN(nn.Module):
    def __init__(self, k, n_class):
        super().__init__()
        self.ec1 = EdgeConv(3, 64)
        self.ec2 = EdgeConv(64, 64)
        self.ec3 = EdgeConv(64, 256)
        self.head = nn.Sequential(
            nn.Linear(64 + 64 + 256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, n_class))

    def forward(self, x):
        x1 = self.ec1(x)
        x2 = self.ec2(x1)
        x3 = self.ec3(x2)
        h = torch.cat([x1, x2, x3], -1).max(1).values
        return self.head(h)


MODELS = {"pointnet": PointNet, "pointnet2_ssg": PointNet2SSG, "dgcnn": DGCNN}


# ---------------- 评估 ----------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        correct += int((model(x).argmax(1) == y).sum())
        total += len(y)
    return correct / total


@torch.no_grad()
def vote_eval(model, ds, rounds=10, batch=64):
    """多次固定种子重采样, 平均 softmax — 点云标准 TTA"""
    model.eval()
    probs = np.zeros((len(ds), ds_nclass), dtype=np.float64)
    for t in range(rounds):
        loader = torch.utils.data.DataLoader(
            CloudSetEval(ds, seed=1000 + t), batch_size=batch)
        pos = 0
        for x, y in loader:
            x = x.to(DEV)
            pr = torch.softmax(model(x), 1).cpu().numpy()
            probs[pos:pos + len(pr)] += pr
            pos += len(pr)
    pred = probs.argmax(1)
    return float((pred == ds.y).mean())


class CloudSetEval(torch.utils.data.Dataset):
    def __init__(self, base_ds, seed):
        self.base, self.seed = base_ds, seed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 100003 + i)
        return self.base._one(i, rng)


ds_nclass = 0  # vote_eval 用的全局


# ---------------- 主流程 ----------------
def main():
    global ds_nclass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dgcnn", choices=list(MODELS))
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--n_points", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--vote", type=int, default=10)
    ap.add_argument("--out", default=str(HERE / "outputs_classify" / "ann_bench"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_points, a.epochs, a.batch, a.vote, a.classes = 128, 2, 16, 2, 10

    t0 = time.time()
    torch.manual_seed(0)
    classes = load_classes(a.classes)
    ds_nclass = len(classes)
    tr = CloudSet(DATA / "modelnet40_train.parquet", classes, a.n_points, train=True)
    te = CloudSet(DATA / "modelnet40_test.parquet", classes, a.n_points, train=False)
    if a.smoke:
        tr.raw, tr.y = tr.raw[:256], tr.y[:256]
        te.raw, te.y = te.raw[:128], te.y[:128]
    print(f"[data] train={len(tr)} test={len(te)} classes={ds_nclass} "
          f"n_points={a.n_points} device={DEV}", flush=True)

    ltr = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True,
                                      num_workers=a.workers, drop_last=True)
    lte = torch.utils.data.DataLoader(te, batch_size=64, num_workers=a.workers)

    model = MODELS[a.model](ds_nclass) if a.model != "dgcnn" else DGCNN(32, ds_nclass)
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lf = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0.0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hist = []
    for ep in range(a.epochs):
        model.train()
        loss_s, nb = 0.0, 0
        for x, y in ltr:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            loss = lf(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_s += float(loss)
            nb += 1
        sched.step()
        acc = evaluate(model, lte)
        hist.append((ep, loss_s / nb, acc))
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), out / f"{a.model}_best.pt")
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"[ep {ep:3d}] loss={loss_s/nb:.4f} test={acc:.4f} best={best:.4f}",
                  flush=True)

    model.load_state_dict(torch.load(out / f"{a.model}_best.pt"))
    vote = vote_eval(model, te, a.vote)
    summary = dict(model=a.model, classes=ds_nclass, n_points=a.n_points,
                   epochs=a.epochs, best_single=best, vote_acc=vote,
                   minutes=round((time.time() - t0) / 60, 2),
                   device=str(DEV), history=hist[-5:])
    with open(out / f"{a.model}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("SUMMARY:", json.dumps({k: v for k, v in summary.items() if k != "history"},
                                 ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
