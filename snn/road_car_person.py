# -*- coding: utf-8 -*-
"""止gap 实验: 本地 ModelNet40 道路二类 (car vs person) 在 road/uav 预设下的分类.

验证"缩小分类对象范围 -> 准确率上升"的定性方向 (~285 训练样本, 趋势级结果).
复用 isal_3dview 的 3D 视角仿真链 (road_crossing: el±5°; uav_cap: el 25-65°).
"""
import json, time
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
DATA = HERE / ".." / "data"
OUT = HERE / "outputs_isal" / "road_car_person"
OUT.mkdir(parents=True, exist_ok=True)

from isal_range_profile import make_scatterers, ESN, ce_readout, cnn1d_acc, N_RNG, SEED
from isal_3dview import profiles_3d, make_dataset, SCEN, D

ROAD = {7: 0, 23: 1}          # ModelNet40 标签 -> 二类索引
NAMES = ["car", "person"]


def load_road_subset(n_points=256, seed=0):
    def prep(parquet):
        df = pd.read_parquet(parquet)
        df = df[df["label"].isin(ROAD)].reset_index(drop=True)
        rng = np.random.default_rng(seed)
        X = np.zeros((len(df), n_points, 3), dtype=np.float32)
        for i, pts in enumerate(df["inputs"].values):
            p = np.stack([np.asarray(q, dtype=np.float32) for q in pts])
            sel = (rng.choice(len(p), n_points, replace=False) if len(p) >= n_points
                   else rng.choice(len(p), n_points, replace=True))
            q = p[sel]
            q -= q.mean(0, keepdims=True)
            q /= (np.abs(q).max() + 1e-8)
            X[i] = q[np.lexsort((q[:, 0], q[:, 1], q[:, 2]))]
        y = np.vectorize(ROAD.get)(df["label"].values).astype(np.int64)
        return X, y
    return prep(DATA / "modelnet40_train.parquet"), prep(DATA / "modelnet40_test.parquet")


def example_fig(X, y, amp, ph0):
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(2, 4, figsize=(12, 4.6), sharex=True)
    for k in (0, 1):
        idx = np.where(y == k)[0][0]
        for j, az in enumerate((0, 45, 90, 135)):
            v = np.array([[np.deg2rad(az), 0.0]])
            p = np.abs(profiles_3d(X[idx], amp[idx], ph0[idx], v, D, rng=rng))[0]
            axes[k, j].plot(p, lw=1.1)
            axes[k, j].set_title(f"{NAMES[k]} az={az}°", fontsize=9)
            axes[k, j].grid(alpha=0.3)
    fig.suptitle("car vs person 单次剖面 (el=0, 赤道)", fontsize=11)
    plt.tight_layout(); plt.savefig(OUT / "profiles_examples.png", dpi=130); plt.close()


def main():
    t0 = time.time()
    (Xtr, ytr), (Xte, yte) = load_road_subset()
    print("[data] train=%d test=%d | car/person: tr %s te %s" %
          (len(Xtr), len(Xte),
           np.bincount(ytr, minlength=2).tolist(),
           np.bincount(yte, minlength=2).tolist()), flush=True)
    amp_tr, ph_tr = make_scatterers(Xtr, SEED)
    amp_te, ph_te = make_scatterers(Xte, SEED + 1)
    example_fig(Xtr, ytr, amp_tr, ph_tr)

    res = {}
    for scenario in SCEN:
        print(f"[{scenario}] ...", flush=True)
        s_tr, q_tr = make_dataset(Xtr, amp_tr, ph_tr, scenario, "scan",
                                  seed_off=0, add_angle=True)
        s_te, q_te = make_dataset(Xte, amp_te, ph_te, scenario, "scan",
                                  seed_off=1, add_angle=True)
        row = {}
        row["单次HRRP+CNN1D"] = cnn1d_acc(s_tr, ytr, s_te, yte, 2, epochs=100)
        esn = ESN(N_RNG + 3, n_res=512)
        row["4波束x3步+ESN+角度"] = ce_readout(
            esn.features(q_tr), ytr, esn.features(q_te), yte, 2, epochs=200)[0]
        res[scenario] = row
        for k, v in row.items():
            print(f"   {k}: {v:.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(SCEN)); w = 0.35
    for j, arm in enumerate(["单次HRRP+CNN1D", "4波束x3步+ESN+角度"]):
        ax.bar(x + j * w, [res[s][arm] for s in SCEN], w, label=arm)
    ax.set_xticks(x + w / 2, list(SCEN))
    ax.axhline(0.5, color="k", ls=":", label="随机(2类)")
    ax.set_ylabel("测试准确率"); ax.set_ylim(0, 1.05)
    ax.set_title("道路二类 (car vs person): 单次 vs 扫描链")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(OUT / "acc_bars.png", dpi=130); plt.close()

    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("done in %.1f s -> %s" % (time.time() - t0, OUT), flush=True)


if __name__ == "__main__":
    main()
