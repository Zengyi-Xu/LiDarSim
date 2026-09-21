# -*- coding: utf-8 -*-
"""参数化道路车辆生成器 (控制变量版) — 不依赖任何外部数据.

7 类道路参与者的几何原型 (米制坐标, 表面采样点云), 尺寸按真实分布抖动:
  sedan / suv / truck / bus / motorcycle / bicycle / pedestrian

与 ModelNet 流程的关键区别: **保留米制尺寸** (不归一化到单位球) ——
道路任务里绝对尺寸本身是判别特征 (货车 12m vs 行人 0.5m).
剖面窗口 D=16m 覆盖最长类 (bus/truck).

实验: road_crossing / uav_cap 预设, 单次 HRRP+CNN1D vs 4 波束扫描+ESN+角度.
"""
import json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
OUT = HERE / "outputs_isal" / "road_vehicles"
OUT.mkdir(parents=True, exist_ok=True)

from isal_range_profile import make_scatterers, ESN, ce_readout, cnn1d_acc, N_RNG, SEED
from isal_3dview import profiles_3d, make_dataset, SCEN

CLASSES = ["sedan", "suv", "truck", "bus", "motorcycle", "bicycle", "pedestrian"]
D_RANGE = 16.0        # 剖面窗口 (米)


# ---------------- 几何原型 ----------------
def box_points(center, size, n, rng):
    """盒体表面均匀采样 n 点"""
    c, s = np.asarray(center), np.asarray(size)
    areas = np.array([s[1] * s[2], s[0] * s[2], s[0] * s[1]]) * 2
    probs = areas / areas.sum()
    counts = rng.multinomial(n, probs)
    pts = []
    for ax, cnt in enumerate(counts):
        others = [i for i in range(3) if i != ax]
        p = rng.uniform(-s[others] / 2, s[others] / 2, (cnt, 2))
        face = rng.choice([-1, 1], cnt) * s[ax] / 2
        q = np.zeros((cnt, 3))
        q[:, others] = p
        q[:, ax] = face
        pts.append(q)
    return np.concatenate(pts) + c


def ring_points(center, radius, n, rng, axis=1):
    """轮子: 圆环 (axis=旋转轴)"""
    t = rng.uniform(0, 2 * np.pi, n)
    p = np.zeros((n, 3))
    axs = [i for i in range(3) if i != axis]
    p[:, axs[0]] = radius * np.cos(t)
    p[:, axs[1]] = radius * np.sin(t)
    p[:, axis] = rng.uniform(-0.05, 0.05, n)
    return p + np.asarray(center)


def gen_object(cls, rng):
    J = lambda a, b: rng.uniform(a, b)
    P = []
    if cls == 0:      # sedan 轿车 4.2-4.9m
        L = J(4.2, 4.9)
        P += [box_points((0, 0, 0.5), (L, 1.8, 1.0), 200, rng),
              box_points((-0.3, 0, 1.25), (L * 0.48, 1.6, 0.5), 80, rng)]
        wx = L * 0.31
        P += [ring_points((sx * wx, sy * 0.8, 0.33), 0.33, 8, rng)
              for sx in (-1, 1) for sy in (-1, 1)]
    elif cls == 1:    # suv
        L = J(4.5, 5.1)
        P += [box_points((0, 0, 0.6), (L, 1.9, 1.2), 220, rng),
              box_points((-0.2, 0, 1.5), (L * 0.5, 1.7, 0.6), 90, rng)]
        wx = L * 0.31
        P += [ring_points((sx * wx, sy * 0.85, 0.36), 0.36, 8, rng)
              for sx in (-1, 1) for sy in (-1, 1)]
    elif cls == 2:    # truck 货车 (车头+货箱)
        Lb = J(7.0, 10.0)
        P += [box_points((Lb / 2 + 1.2, 0, 1.6), (2.2, 2.3, 3.2), 90, rng),
              box_points((-1.0, 0, 1.6), (Lb, 2.4, 3.0), 260, rng)]
        for wx in (Lb / 2 + 1.0, -Lb / 4, -Lb / 2):
            P += [ring_points((wx, sy * 1.1, 0.5), 0.5, 8, rng) for sy in (-1, 1)]
    elif cls == 3:    # bus 巴士
        L = J(10.0, 12.5)
        P += [box_points((0, 0, 1.5), (L, 2.5, 2.6), 320, rng)]
        for wx in (L * 0.33, -L * 0.33):
            P += [ring_points((wx, sy * 1.1, 0.5), 0.5, 8, rng) for sy in (-1, 1)]
    elif cls == 4:    # motorcycle 摩托+骑手
        P += [box_points((0, 0, 0.7), (J(1.8, 2.1), 0.7, 0.5), 90, rng),
              box_points((-0.1, 0, 1.35), (0.45, 0.45, 0.8), 60, rng),
              ring_points((0.75, 0, 0.35), 0.35, 10, rng),
              ring_points((-0.75, 0, 0.35), 0.35, 10, rng)]
    elif cls == 5:    # bicycle 自行车+骑手
        P += [box_points((0, 0, 0.6), (J(1.6, 1.9), 0.4, 0.3), 50, rng),
              box_points((0, 0, 1.4), (0.45, 0.5, 0.9), 70, rng),
              ring_points((0.7, 0, 0.35), 0.35, 10, rng),
              ring_points((-0.7, 0, 0.35), 0.35, 10, rng)]
    else:             # pedestrian 行人
        P += [box_points((0, 0, 1.15), (0.5, 0.35, 0.7), 60, rng),
              box_points((0.12, 0, 0.5), (0.16, 0.2, 1.0), 30, rng),
              box_points((-0.12, 0, 0.5), (0.16, 0.2, 1.0), 30, rng),
              box_points((0, 0, 1.75), (0.22, 0.22, 0.25), 20, rng)]
    return np.concatenate(P).astype(np.float32)


def gen_dataset(n_per, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(len(CLASSES)):
        for _ in range(n_per):
            X.append(gen_object(c, rng))
            y.append(c)
    return X, np.array(y, dtype=np.int64)


def to_profile_coords(pts):
    """米制 -> profiles_3d 的归一化坐标 (P*D/2 = 真实米)"""
    return pts / (D_RANGE / 2.0)


# ---------------- 图 ----------------
def example_fig(X, y):
    fig = plt.figure(figsize=(13, 6.5))
    rng = np.random.default_rng(3)
    for k in range(7):
        idx = np.where(y == k)[0][0]
        ax = fig.add_subplot(2, 4, k + 1, projection="3d")
        p = X[idx]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.5, c="C0")
        ax.set_title(CLASSES[k], fontsize=10)
        ax.set_box_aspect([max(4, np.ptp(p[:, 0])), max(2, np.ptp(p[:, 1]) + 0.1),
                           max(2, np.ptp(p[:, 2]) + 0.1)])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle("参数化道路车辆原型 (米制)", fontsize=12)
    plt.tight_layout(); plt.savefig(OUT / "vehicle_prototypes.png", dpi=130); plt.close()


def size_fig(X, y):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    spans = np.array([np.ptp(x[:, 0]) for x in X])
    for c in range(7):
        ax.hist(spans[y == c], bins=30, histtype="step", lw=1.5, label=CLASSES[c])
    ax.set(xlabel="长度 span (m)", ylabel="频数", title="各类长度分布 (判别特征)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "size_distribution.png", dpi=130); plt.close()


# ---------------- 主实验 ----------------
def resample(X, n=300, seed=0):
    """变长点云重采样到固定 n 点 (少量类上采样)"""
    rng = np.random.default_rng(seed)
    out = np.zeros((len(X), n, 3), dtype=np.float32)
    for i, p in enumerate(X):
        sel = rng.choice(len(p), n, replace=len(p) < n)
        out[i] = p[sel]
    return out


def main():
    t0 = time.time()
    n_tr, n_te = 350, 100
    Xtr, ytr = gen_dataset(n_tr, SEED)
    Xte, yte = gen_dataset(n_te, SEED + 100)
    Xtr = resample([to_profile_coords(x) for x in Xtr])
    Xte = resample([to_profile_coords(x) for x in Xte])
    print("[data] train=%d test=%d, 7 classes, D=%.0fm" %
          (len(Xtr), len(Xte), D_RANGE), flush=True)
    amp_tr, ph_tr = make_scatterers(Xtr, SEED)
    amp_te, ph_te = make_scatterers(Xte, SEED + 1)
    example_fig(Xtr, ytr)
    size_fig([x * (D_RANGE / 2) for x in Xtr], ytr)

    res = {}
    for scenario in SCEN:
        print(f"[{scenario}] ...", flush=True)
        s_tr, q_tr = make_dataset(Xtr, amp_tr, ph_tr, scenario, "scan",
                                  seed_off=0, add_angle=True)
        s_te, q_te = make_dataset(Xte, amp_te, ph_te, scenario, "scan",
                                  seed_off=1, add_angle=True)
        row = {}
        row["单次HRRP+CNN1D"] = cnn1d_acc(s_tr, ytr, s_te, yte, 7, epochs=100)
        esn = ESN(N_RNG + 3, n_res=512)
        row["4波束x3步+ESN+角度"] = ce_readout(
            esn.features(q_tr), ytr, esn.features(q_te), yte, 7, epochs=200)[0]
        res[scenario] = row
        for k, v in row.items():
            print(f"   {k}: {v:.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(SCEN)); w = 0.35
    for j, arm in enumerate(["单次HRRP+CNN1D", "4波束x3步+ESN+角度"]):
        ax.bar(x + j * w, [res[s][arm] for s in SCEN], w, label=arm)
    ax.set_xticks(x + w / 2, list(SCEN))
    ax.axhline(1 / 7, color="k", ls=":", label="随机(7类)")
    ax.set_ylabel("测试准确率"); ax.set_ylim(0, 1.05)
    ax.set_title("参数化道路 7 类: 单次 vs 扫描链")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(OUT / "acc_bars.png", dpi=130); plt.close()

    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("done in %.1f s -> %s" % (time.time() - t0, OUT), flush=True)


if __name__ == "__main__":
    main()
