# -*- coding: utf-8 -*-
"""
SNN/LSM classification demo on real datasets.

Goal: verify that a spiking reservoir (LSM) can do *classification* tasks,
not just radar detection. Two datasets:

  1. sklearn digits (8x8 MNIST proxy): pixel intensity -> spike rate.
  2. ModelNet40 point cloud subset: two spike encodings —
     a. rectified coordinate rate coding (x+/-, y+/-, z+/- channels)
     b. range-echo coding (physically motivated: spike time ~ radial distance,
        emulating what the optical grating-compression front-end would produce)

Readout: ridge regression (one-vs-rest) on LSM [mean, traj] features.
Baselines: logistic regression on raw features; chance level.
"""
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reservoir import LSM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs_classify")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# readout
# ---------------------------------------------------------------------------

def ridge_readout(Xtr, ytr, Xte, yte, n_classes, alpha=10.0):
    """One-vs-rest ridge regression -> predicted class = argmax score.
    Features are standardized first (scales differ wildly across nodes).
    Uses the dual (sample-space) form when d >> n to avoid huge matrices."""
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-12
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    n, d = Xtr.shape
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), ytr] = 1.0
    Xb = np.hstack([Xtr, np.ones((n, 1))])
    Xte_b = np.hstack([Xte, np.ones((len(Xte), 1))])
    if d + 1 <= n:
        A = Xb.T @ Xb + alpha * np.eye(d + 1)
        W = np.linalg.solve(A, Xb.T @ Y)
    else:
        # dual form: W = Xb^T (Xb Xb^T + alpha I)^-1 Y
        K = Xb @ Xb.T + alpha * np.eye(n)
        W = Xb.T @ np.linalg.solve(K, Y)
    scores = Xte_b @ W
    pred = scores.argmax(axis=1)
    return (pred == yte).mean(), pred


def lsm_features(spikes, lsm):
    mean, traj = lsm.run(spikes)
    return np.concatenate([mean.numpy(), traj.numpy()], axis=1)


# ---------------------------------------------------------------------------
# dataset 1: sklearn digits (MNIST proxy)
# ---------------------------------------------------------------------------

def run_digits(n_train=1200, n_test=400, T=60, n_res=256, seed=0):
    from sklearn.datasets import load_digits
    print("\n[digits 8x8, %d train / %d test]" % (n_train, n_test))
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    tr, te = perm[:n_train], perm[n_train:n_train + n_test]
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]

    # rate coding: 64 pixel channels, Bernoulli spikes over T steps
    def encode(Xa):
        rates = np.repeat(Xa[:, :, None], T, axis=2)     # (B, 64, T)
        return torch.tensor(
            (np.random.default_rng(seed).random(rates.shape) < rates * 0.9),
            dtype=torch.float32)

    lsm = LSM(n_in=64, n_res=n_res, seed=seed)
    Ftr = lsm_features(encode(Xtr), lsm)
    Fte = lsm_features(encode(Xte), lsm)
    acc, pred = ridge_readout(Ftr, ytr, Fte, yte, 10)

    # baselines
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=3000).fit(Xtr, ytr)
    raw_acc = clf.score(Xte, yte)

    print("  LSM reservoir accuracy: %.3f" % acc)
    print("  raw-pixel logistic baseline: %.3f" % raw_acc)
    return {"acc": acc, "raw_acc": raw_acc, "y_test": yte, "pred": pred,
            "spike_rate": lsm.last_spike_rate}


# ---------------------------------------------------------------------------
# dataset 2: ModelNet40 point cloud
# ---------------------------------------------------------------------------

CLASSES10 = ["airplane", "car", "chair", "table", "bottle",
             "lamp", "sofa", "airplane", "guitar", "laptop"]  # will fix from data


def load_modelnet_subset(n_classes=10, n_points=128, seed=0):
    import pandas as pd
    df = pd.read_parquet(os.path.join(HERE, "..", "data", "modelnet40_train.parquet"))
    dft = pd.read_parquet(os.path.join(HERE, "..", "data", "modelnet40_test.parquet"))
    rng = np.random.default_rng(seed)
    classes = np.sort(df["label"].unique())[:n_classes]
    df = df[df["label"].isin(classes)]
    dft = dft[dft["label"].isin(classes)]

    def prep(df):
        X = np.zeros((len(df), n_points, 3))
        for i, pts in enumerate(df["inputs"].values):
            pts = np.stack([np.asarray(p, dtype=np.float32) for p in pts])
            if len(pts) >= n_points:
                sel = rng.choice(len(pts), n_points, replace=False)
            else:
                sel = rng.choice(len(pts), n_points, replace=True)
            p = pts[sel]
            p -= p.mean(axis=0, keepdims=True)
            p /= (np.abs(p).max() + 1e-8)
            # canonical sort so the reservoir sees a consistent order
            key = np.lexsort((p[:, 0], p[:, 1], p[:, 2]))
            X[i] = p[key]
        return X, df["label"].to_numpy()

    return prep(df), prep(dft), classes


def encode_coords(X, T=None, max_rate=0.9, seed=0):
    """Rectified coordinate rate coding: 6 channels (x+/-, y+/-, z+/-),
    time step = point index (canonical order)."""
    B, N, _ = X.shape
    T = T or N
    ch = np.stack([np.clip(X[:, :, 0], 0, None), np.clip(-X[:, :, 0], 0, None),
                   np.clip(X[:, :, 1], 0, None), np.clip(-X[:, :, 1], 0, None),
                   np.clip(X[:, :, 2], 0, None), np.clip(-X[:, :, 2], 0, None)],
                  axis=2)                                    # (B, N, 6)
    ch = np.transpose(ch, (0, 2, 1))                         # (B, 6, N)
    rates = np.repeat(ch[:, :, :, None], 1, axis=3).reshape(B, 6, N)
    rng = np.random.default_rng(seed)
    return torch.tensor((rng.random((B, 6, N)) < rates * max_rate),
                        dtype=torch.float32)


def encode_range_echo(X, n_bins=96, seed=0):
    """Physically motivated: spike time ~ radial distance of the point.
    This is what a grating-compression LiDAR front-end would emit.
    Channels = 3 view directions (xy, yz, azimuth-split), time = range bin."""
    B = X.shape[0]
    r = np.linalg.norm(X, axis=2)                            # (B, N)
    r = r / (r.max(axis=1, keepdims=True) + 1e-8)
    spikes = np.zeros((B, 3, n_bins), dtype=np.float32)
    for c in range(3):                                       # 3 angular sectors
        mask = (X[:, :, c % 3] > -0.33) if c < 2 else np.ones(X.shape[1], bool)
        for b in range(B):
            m = mask if c >= 2 else mask[b]
            idx = np.floor(r[b][m] * (n_bins - 1)).astype(int)
            np.add.at(spikes[b, c], idx, 1.0)
    spikes = spikes / (spikes.max(axis=2, keepdims=True) + 1e-8)
    return torch.tensor(spikes > 0.5, dtype=torch.float32)


def run_modelnet(n_classes=10, n_train_per=150, n_test_per=40, n_points=256,
                 n_res=512, seed=0):
    print("\n[ModelNet40 subset: %d classes, %d/%d per class, %d points]" %
          (n_classes, n_train_per, n_test_per, n_points))
    (Xtr_all, ytr_all), (Xte_all, yte_all), classes = load_modelnet_subset(
        n_classes, n_points, seed)

    rng = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    for c in classes:
        idx_tr = np.where(ytr_all == c)[0]
        idx_te = np.where(yte_all == c)[0]
        tr_idx.append(rng.choice(idx_tr, min(n_train_per, len(idx_tr)), replace=False))
        te_idx.append(rng.choice(idx_te, min(n_test_per, len(idx_te)), replace=False))
    tr_idx = np.concatenate(tr_idx)
    te_idx = np.concatenate(te_idx)
    Xtr, ytr = Xtr_all[tr_idx], ytr_all[tr_idx]
    Xte, yte = Xte_all[te_idx], yte_all[te_idx]
    label_map = {c: i for i, c in enumerate(classes)}
    ytr = np.vectorize(label_map.get)(ytr)
    yte = np.vectorize(label_map.get)(yte)

    # encoding A: rectified coordinates
    lsmA = LSM(n_in=6, n_res=n_res, seed=seed)
    FAtr = lsm_features(encode_coords(Xtr, seed=seed), lsmA)
    FAte = lsm_features(encode_coords(Xte, seed=seed + 1), lsmA)
    accA, predA = ridge_readout(FAtr, ytr, FAte, yte, len(classes))

    # encoding B: range echo
    lsmB = LSM(n_in=3, n_res=n_res, seed=seed)
    FBtr = lsm_features(encode_range_echo(Xtr, seed=seed), lsmB)
    FBte = lsm_features(encode_range_echo(Xte, seed=seed + 1), lsmB)
    accB, predB = ridge_readout(FBtr, ytr, FBte, yte, len(classes))

    # combined encoding features
    FCtr = np.concatenate([FAtr, FBtr], axis=1)
    FCte = np.concatenate([FAte, FBte], axis=1)
    accC, predC = ridge_readout(FCtr, ytr, FCte, yte, len(classes))

    # baseline: LR on sorted coordinates (flattened)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=5000).fit(
        Xtr.reshape(len(Xtr), -1), ytr)
    raw_acc = clf.score(Xte.reshape(len(Xte), -1), yte)

    print("  LSM coord-rate encoding accuracy: %.3f" % accA)
    print("  LSM range-echo encoding accuracy: %.3f" % accB)
    print("  LSM combined encoding accuracy:   %.3f" % accC)
    print("  raw-coordinate logistic baseline: %.3f" % raw_acc)
    return {"accA": accA, "accB": accB, "accC": accC, "raw_acc": raw_acc,
            "y_test": yte, "predA": predA, "predB": predB, "predC": predC,
            "classes": classes, "n_train": len(Xtr), "n_test": len(Xte)}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def confusion_figure(y_true, pred, title, fname, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for a, b in zip(y_true, pred):
        cm[a, b] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def summary_figure(res_digits, res_mn):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["digits\nLSM", "digits\nraw-LR",
              "ModelNet\nLSM-coord", "ModelNet\nLSM-echo", "ModelNet\nLSM-combined",
              "ModelNet\nraw-LR"]
    vals = [res_digits["acc"], res_digits["raw_acc"],
            res_mn["accA"], res_mn["accB"], res_mn["accC"], res_mn["raw_acc"]]
    colors = ["#2a9d8f", "#999999", "#2a9d8f", "#264653", "#e76f51", "#999999"]
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, "%.2f" % v, ha="center")
    ax.axhline(0.1, color="k", ls="--", alpha=0.4, label="chance (10 classes)")
    ax.set_ylabel("test accuracy")
    ax.set_title("SNN reservoir classification on real datasets")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "classification_summary.png"), dpi=150)
    plt.close(fig)
    print("saved classification_summary.png")


def run_modelnet_scaling(seed=0):
    """Accuracy vs reservoir size and training-set size (combined encoding).
    Documents the expressivity ceiling of a fixed random reservoir."""
    print("\n[ModelNet40 scaling sweep]")
    (Xtr_all, ytr_all), (Xte_all, yte_all), classes = load_modelnet_subset(
        10, 256, seed)
    rng = np.random.default_rng(seed)
    label_map = {c: i for i, c in enumerate(classes)}

    def subset(n_train_per, n_test_per=40):
        tr_idx, te_idx = [], []
        for c in classes:
            idx_tr = np.where(ytr_all == c)[0]
            idx_te = np.where(yte_all == c)[0]
            tr_idx.append(rng.choice(idx_tr, min(n_train_per, len(idx_tr)), replace=False))
            te_idx.append(rng.choice(idx_te, min(n_test_per, len(idx_te)), replace=False))
        tr_idx = np.concatenate(tr_idx)
        te_idx = np.concatenate(te_idx)
        return (Xtr_all[tr_idx], np.vectorize(label_map.get)(ytr_all[tr_idx]),
                Xte_all[te_idx], np.vectorize(label_map.get)(yte_all[te_idx]))

    # sweep reservoir size at fixed train size (average over reservoir seeds)
    n_res_list = [64, 128, 256, 512, 1024]
    seeds = [0, 1, 2]
    res_accs = []
    res_accs_std = []
    Xtr, ytr, Xte, yte = subset(150)
    for n_res in n_res_list:
        accs_seeds = []
        for sd_ in seeds:
            lsmA = LSM(n_in=6, n_res=n_res, seed=sd_)
            FAtr = lsm_features(encode_coords(Xtr, seed=sd_), lsmA)
            FAte = lsm_features(encode_coords(Xte, seed=sd_ + 100), lsmA)
            lsmB = LSM(n_in=3, n_res=n_res, seed=sd_)
            FBtr = lsm_features(encode_range_echo(Xtr, seed=sd_), lsmB)
            FBte = lsm_features(encode_range_echo(Xte, seed=sd_ + 100), lsmB)
            a, _ = ridge_readout(np.concatenate([FAtr, FBtr], axis=1), ytr,
                                 np.concatenate([FAte, FBte], axis=1), yte, len(classes))
            accs_seeds.append(a)
        res_accs.append(np.mean(accs_seeds))
        res_accs_std.append(np.std(accs_seeds))
        print("  n_res=%d: acc=%.3f +/- %.3f" % (n_res, res_accs[-1], res_accs_std[-1]))

    # sweep training-set size at fixed reservoir size
    train_sizes = [20, 40, 80, 150, 300]
    train_accs = []
    train_accs_std = []
    for ntp in train_sizes:
        accs_seeds = []
        for sd_ in seeds:
            Xtr, ytr, Xte, yte = subset(ntp)
            n_res = 512
            lsmA = LSM(n_in=6, n_res=n_res, seed=sd_)
            FAtr = lsm_features(encode_coords(Xtr, seed=sd_), lsmA)
            FAte = lsm_features(encode_coords(Xte, seed=sd_ + 100), lsmA)
            lsmB = LSM(n_in=3, n_res=n_res, seed=sd_)
            FBtr = lsm_features(encode_range_echo(Xtr, seed=sd_), lsmB)
            FBte = lsm_features(encode_range_echo(Xte, seed=sd_ + 100), lsmB)
            a, _ = ridge_readout(np.concatenate([FAtr, FBtr], axis=1), ytr,
                                 np.concatenate([FAte, FBte], axis=1), yte, len(classes))
            accs_seeds.append(a)
        train_accs.append(np.mean(accs_seeds))
        train_accs_std.append(np.std(accs_seeds))
        print("  n_train_per=%d: acc=%.3f +/- %.3f" % (ntp, train_accs[-1], train_accs_std[-1]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.errorbar(n_res_list, res_accs, yerr=res_accs_std, fmt="o-", lw=1.5,
                color="#2a9d8f", capsize=4)
    ax.set_xscale("log")
    ax.axhline(0.1, color="k", ls="--", alpha=0.4)
    ax.set_xlabel("reservoir size (neurons)")
    ax.set_ylabel("test accuracy")
    ax.set_title("accuracy vs reservoir size (combined enc.)")
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.errorbar(train_sizes, train_accs, yerr=train_accs_std, fmt="s-", lw=1.5,
                color="#e76f51", capsize=4)
    ax.set_xscale("log")
    ax.axhline(0.1, color="k", ls="--", alpha=0.4)
    ax.set_xlabel("training samples per class")
    ax.set_ylabel("test accuracy")
    ax.set_title("accuracy vs training-set size (n_res=512)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "modelnet_scaling.png"), dpi=150)
    plt.close(fig)
    print("  saved modelnet_scaling.png")
    return {"n_res_list": n_res_list, "res_accs": res_accs, "res_accs_std": res_accs_std,
            "train_sizes": train_sizes, "train_accs": train_accs,
            "train_accs_std": train_accs_std}


def main():
    t0 = time.time()
    res_digits = run_digits()
    res_mn = run_modelnet()
    res_scaling = run_modelnet_scaling()

    confusion_figure(res_digits["y_test"], res_digits["pred"],
                     "digits: LSM reservoir confusion", "confusion_digits.png")
    confusion_figure(res_mn["y_test"], res_mn["predC"],
                     "ModelNet40 subset: LSM combined-encoding confusion",
                     "confusion_modelnet.png")
    summary_figure(res_digits, res_mn)

    np.savez(os.path.join(OUT, "classification_results.npz"),
             digits_acc=res_digits["acc"], digits_raw=res_digits["raw_acc"],
             digits_spike_rate=res_digits["spike_rate"],
             mn_acc_coord=res_mn["accA"], mn_acc_echo=res_mn["accB"],
             mn_acc_combined=res_mn["accC"], mn_raw=res_mn["raw_acc"],
             mn_n_train=res_mn["n_train"], mn_n_test=res_mn["n_test"],
             scaling_n_res=res_scaling["n_res_list"],
             scaling_res_accs=res_scaling["res_accs"],
             scaling_train_sizes=res_scaling["train_sizes"],
             scaling_train_accs=res_scaling["train_accs"])
    print("\nfinished in %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
