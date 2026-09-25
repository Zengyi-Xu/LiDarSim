# -*- coding: utf-8 -*-
"""
M7: 信息量匹配条件下的分类对比（KGFP proposal 预备数据）

实验矩阵（统一: ModelNet40 前10类, 256点, 150/类训练, 40/类测试, 种子0/1/2）：
  E1 range-echo + LSM + ridge   （复现 M2 下界, 对照）
  E2 range-echo + LSM + MLP     （主数字: 物理合理格式 + 训练读出）
  E3 坐标速率编码 + LSM + MLP   （脉冲编码上限）
  E4 原始浮点坐标 + MLP          （信息量天花板, 对照 E2 量化编码损失）
  E5 原始浮点坐标 + ridge        （M2 基线的加强版对照）

消融（E2 配置）：
  A1 bins 48/96/192;  A2 n_res 256/512/1024;
  A3 读出 ridge/MLP-1/MLP-2;  A4 n_train_per 75/150/300
鲁棒性（E2 配置）：
  R1 池权重量化 4/5/6/8 bit（对应 MRR crossbar 6.74 bit）;
  R2 输入脉冲抖动（时间 bin 偏移 + 翻转噪声）

用法：
  python snn/m7_matched_info.py --smoke   # 冒烟: 1 种子, n_res 256
  python snn/m7_matched_info.py           # 全量
"""
import os
import sys
import json
import time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from reservoir import LSM
from classification_demo import (load_modelnet_subset, encode_coords,
                                 encode_range_echo, ridge_readout)

OUT = os.path.join(ROOT, "outputs_m7")
os.makedirs(OUT, exist_ok=True)

N_CLASSES, N_POINTS = 10, 256
SMOKE = "--smoke" in sys.argv
SEEDS = [0] if SMOKE else [0, 1, 2]


def subsample(X, y, classes, per_class, seed):
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y == c)[0],
                                     min(per_class, (y == c).sum()),
                                     replace=False)
                          for c in classes])
    lab = {c: i for i, c in enumerate(classes)}
    return X[idx], np.vectorize(lab.get)(y[idx])


def mlp_readout(Xtr, ytr, Xte, yte, n_classes, hidden=(256,), epochs=100,
                seed=0):
    """标准化 + 全连接 MLP + 交叉熵，val 早停选模。返回 test acc。"""
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-12
    Xtr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    Xte = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    n_val = len(ytr) // 5
    g = np.random.default_rng(seed)
    perm = g.permutation(len(ytr))
    vi, ti = perm[:n_val], perm[n_val:]
    layers, din = [], Xtr.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(din, h), torch.nn.ReLU()]
        din = h
    layers.append(torch.nn.Linear(din, n_classes))
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = torch.nn.CrossEntropyLoss()
    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        lf(net(Xtr[ti]), ytr_t[ti]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va = (net(Xtr[vi]).argmax(1) == ytr_t[vi]).float().mean().item()
        if va > best_acc:
            best_acc = va
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        acc = (net(Xte).argmax(1) == torch.tensor(yte)).float().mean().item()
    return acc


def lsm_feats(spikes, n_res, seed):
    lsm = LSM(n_in=spikes.shape[1], n_res=n_res, seed=seed)
    mean, traj = lsm.run(spikes)
    return np.concatenate([mean.numpy(), traj.numpy()], axis=1), lsm


def quantize(t, bits):
    s = t.abs().max() + 1e-12
    q = 2 ** (bits - 1)
    return (torch.round(t / s * q) / q) * s


def jitter_spikes(spk, shift_std=0.0, flip_p=0.0, seed=0):
    """R2: 时间 bin 抖动（零填充移位，不回绕）+ 随机翻转"""
    rng = np.random.default_rng(seed)
    s = spk.clone()
    if shift_std > 0:
        shifts = np.round(rng.normal(0, shift_std, s.shape[:2])).astype(int)
        out = torch.zeros_like(s)
        T = s.shape[2]
        for b in range(s.shape[0]):
            for c in range(s.shape[1]):
                k = shifts[b, c]
                if k == 0:
                    out[b, c] = s[b, c]
                elif k > 0:
                    out[b, c, k:] = s[b, c, :T - k]
                else:
                    out[b, c, :T + k] = s[b, c, -k:]
        s = out
    if flip_p > 0:
        fl = torch.tensor(rng.random(s.shape) < flip_p, dtype=torch.float32)
        s = torch.clamp(s + fl * (1 - 2 * s), 0, 1)
    return s


def encode_range_echo_analog(X, n_bins=96):
    """range-echo 直方图归一化，不做硬门限（模拟光电流驱动）"""
    B = X.shape[0]
    r = np.linalg.norm(X, axis=2)
    r = r / (r.max(axis=1, keepdims=True) + 1e-8)
    out = np.zeros((B, 3, n_bins), dtype=np.float32)
    for c in range(3):
        mask = (X[:, :, c % 3] > -0.33) if c < 2 else np.ones(X.shape[1], bool)
        for b in range(B):
            m = mask if c >= 2 else mask[b]
            idx = np.floor(r[b][m] * (n_bins - 1)).astype(int)
            np.add.at(out[b, c], idx, 1.0)
    out = out / (out.max(axis=2, keepdims=True) + 1e-8)
    return torch.tensor(out, dtype=torch.float32)


def encode_coords_analog(X, seed=0):
    """整流坐标直接作速率（6 通道 × 256 步，不泊松采样）"""
    B, N, _ = X.shape
    ch = np.stack([np.clip(X[:, :, 0], 0, None), np.clip(-X[:, :, 0], 0, None),
                   np.clip(X[:, :, 1], 0, None), np.clip(-X[:, :, 1], 0, None),
                   np.clip(X[:, :, 2], 0, None), np.clip(-X[:, :, 2], 0, None)],
                  axis=2)
    return torch.tensor(np.transpose(ch, (0, 2, 1)), dtype=torch.float32)


def run_config(Xtr, ytr, Xte, yte, mode, n_res=512, n_bins=96,
               readout="mlp1", n_train_per=150, seed=0,
               quant_bits=None, jit_shift=0.0, jit_flip=0.0):
    """mode: 'echo' | 'coord' | 'raw'"""
    if mode == "raw":
        Ftr = Xtr.reshape(len(Xtr), -1)
        Fte = Xte.reshape(len(Xte), -1)
    else:
        enc = {"echo": encode_range_echo, "coord": encode_coords,
               "echo_analog": encode_range_echo_analog,
               "coord_analog": encode_coords_analog}[mode]
        kw = {"n_bins": n_bins} if mode.startswith("echo") else {}
        spk_tr = enc(Xtr, seed=seed, **kw) if "analog" not in mode else enc(Xtr, **kw)
        spk_te = enc(Xte, seed=seed + 1, **kw) if "analog" not in mode else enc(Xte, **kw)
        if jit_shift or jit_flip:
            spk_tr = jitter_spikes(spk_tr, jit_shift, jit_flip, seed)
            spk_te = jitter_spikes(spk_te, jit_shift, jit_flip, seed + 1)
        Ftr, lsm = lsm_feats(spk_tr, n_res, seed)
        if quant_bits:
            lsm.W_rec = quantize(lsm.W_rec, quant_bits)
            lsm.W_in = quantize(lsm.W_in, quant_bits)
            Ftr = np.concatenate([lsm.run(spk_tr)[0].numpy(),
                                  lsm.run(spk_tr)[1].numpy()], 1)
            Fte = np.concatenate([lsm.run(spk_te)[0].numpy(),
                                  lsm.run(spk_te)[1].numpy()], 1)
        else:
            Fte = np.concatenate([lsm.run(spk_te)[0].numpy(),
                                  lsm.run(spk_te)[1].numpy()], 1)
    if readout == "ridge":
        acc, _ = ridge_readout(Ftr, ytr, Fte, yte, N_CLASSES)
    else:
        hidden = (256,) if readout == "mlp1" else (256, 128)
        acc = mlp_readout(Ftr, ytr, Fte, yte, N_CLASSES, hidden=hidden,
                          seed=seed)
    return acc


def main():
    t0 = time.time()
    (Xtr_all, ytr_all), (Xte_all, yte_all), classes = load_modelnet_subset(
        N_CLASSES, N_POINTS, seed=0)
    results = {}

    def run_block(name, specs):
        accs = []
        for seed in SEEDS:
            Xtr, ytr = subsample(Xtr_all, ytr_all, classes,
                                 specs.get("n_train_per", 150), seed)
            Xte, yte = subsample(Xte_all, yte_all, classes, 40, seed)
            kw = {k: v for k, v in specs.items() if k != "n_train_per"}
            accs.append(run_config(Xtr, ytr, Xte, yte, seed=seed, **kw))
            print("  %s seed=%d: %.3f" % (name, seed, accs[-1]), flush=True)
        results[name] = {"accs": accs,
                         "mean": float(np.mean(accs)), "std": float(np.std(accs))}
        print("%s: %.3f ± %.3f" % (name, results[name]["mean"],
                                   results[name]["std"]), flush=True)

    print("===== E1-E5 主矩阵 =====")
    run_block("E1_echo_ridge",  dict(mode="echo", readout="ridge"))
    run_block("E2_echo_mlp",    dict(mode="echo", readout="mlp1"))
    run_block("E2b_echoA_mlp",  dict(mode="echo_analog", readout="mlp1"))
    run_block("E3_coord_mlp",   dict(mode="coord", readout="mlp1"))
    run_block("E3b_coordA_mlp", dict(mode="coord_analog", readout="mlp1"))
    run_block("E4_raw_mlp",     dict(mode="raw", readout="mlp1"))
    run_block("E5_raw_ridge",   dict(mode="raw", readout="ridge"))
    # sklearn LR 基线（与 M2 同法，直接算，不经 run_config）
    from sklearn.linear_model import LogisticRegression
    accs = []
    for seed in SEEDS:
        Xtr, ytr = subsample(Xtr_all, ytr_all, classes, 150, seed)
        Xte, yte = subsample(Xte_all, yte_all, classes, 40, seed)
        clf = LogisticRegression(max_iter=5000).fit(
            Xtr.reshape(len(Xtr), -1), ytr)
        accs.append(clf.score(Xte.reshape(len(Xte), -1), yte))
    results["E5b_raw_lr"] = {"accs": accs, "mean": float(np.mean(accs)),
                             "std": float(np.std(accs))}
    print("E5b_raw_lr: %.3f ± %.3f" % (np.mean(accs), np.std(accs)), flush=True)

    print("===== 消融 A1 编码分辨率 =====")
    for b in (48, 192):
        run_block("A1_bins%d" % b, dict(mode="echo", n_bins=b, readout="mlp1"))
    print("===== 消融 A2 池规模 =====")
    for nr in (256, 1024):
        run_block("A2_nres%d" % nr, dict(mode="echo", n_res=nr, readout="mlp1"))
    print("===== 消融 A3 读出容量 =====")
    run_block("A3_mlp2", dict(mode="echo", readout="mlp2"))
    print("===== 消融 A4 训练量 =====")
    for ntr in (75, 300):
        run_block("A4_ntrain%d" % ntr,
                  dict(mode="echo", readout="mlp1", n_train_per=ntr))

    print("===== 鲁棒性 R1 权重量化 =====")
    for qb in (4, 5, 6, 8):
        run_block("R1_quant%dbit" % qb,
                  dict(mode="echo", readout="mlp1", quant_bits=qb))
    print("===== 鲁棒性 R2 输入抖动/噪声 =====")
    for js, jf in ((1.0, 0.01), (2.0, 0.05)):
        run_block("R2_shift%.0f_flip%.2f" % (js, jf),
                  dict(mode="echo", readout="mlp1", jit_shift=js, jit_flip=jf))

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved outputs_m7/results.json  (total %.1f min)"
          % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
