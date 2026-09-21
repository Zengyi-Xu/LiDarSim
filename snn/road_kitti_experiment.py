# -*- coding: utf-8 -*-
"""KITTI 道路对象库分类实验 (待 road_objects.npz 就绪后运行).

流程: road_object_extractor.py 产出的对象点云 -> 散射体 -> HRRP/扫描链
(road_crossing / uav_cap 预设), 单次 vs 4 波束扫描+ESN+角度.

与参数化版 (road_vehicles.py) 的差别: 点云来自真实激光雷达观测 ——
稀疏度/遮挡/截断真实, 是"信息受限"故事的最诚实测试.

用法:
  python road_object_extractor.py --root <KITTI>/training   # 先提取
  python road_kitti_experiment.py                           # 再跑本实验
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
NPZ = HERE / ".." / "data" / "road_objects.npz"
OUT = HERE / "outputs_isal" / "road_kitti"
OUT.mkdir(parents=True, exist_ok=True)

from isal_range_profile import make_scatterers, ESN, ce_readout, cnn1d_acc, N_RNG, SEED
from isal_3dview import make_dataset, SCEN
from road_vehicles import resample, to_profile_coords, D_RANGE

# 类合并: car+van -> car, 其余保留 (van/tram 样本少时并入 car)
MERGE = {"car": 0, "van": 0, "truck": 1, "pedestrian": 2, "cyclist": 3}
NAMES = ["car", "truck", "pedestrian", "cyclist"]


def load_library():
    d = np.load(NPZ, allow_pickle=True)
    names = [str(x) for x in d["class_names"]]
    P, L = [], []
    for pts, lab in zip(d["points"], d["labels"]):
        nm = names[int(lab)]
        if nm in MERGE:
            P.append(np.asarray(pts, dtype=np.float32))
            L.append(MERGE[nm])
    return P, np.array(L, dtype=np.int64)


def split(P, y, n_tr_per=300, seed=SEED):
    rng = np.random.default_rng(seed)
    tr_i, te_i = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_tr = min(n_tr_per, int(len(idx) * 0.75))
        tr_i += idx[:n_tr].tolist()
        te_i += idx[n_tr:].tolist()
    return tr_i, te_i


def main():
    assert NPZ.exists(), ("先运行 road_object_extractor.py 生成 " + str(NPZ))
    t0 = time.time()
    P, y = load_library()
    print("[lib] objects=%d, class counts=%s" %
          (len(P), np.bincount(y, minlength=len(NAMES)).tolist()), flush=True)
    tr_i, te_i = split(P, y)
    Xtr = resample([to_profile_coords(P[i] - P[i].mean(0)) for i in tr_i])
    Xte = resample([to_profile_coords(P[i] - P[i].mean(0)) for i in te_i])
    ytr, yte = y[tr_i], y[te_i]
    print("[data] train=%d test=%d" % (len(Xtr), len(Xte)), flush=True)
    amp_tr, ph_tr = make_scatterers(Xtr, SEED)
    amp_te, ph_te = make_scatterers(Xte, SEED + 1)

    res = {}
    for scenario in SCEN:
        print(f"[{scenario}] ...", flush=True)
        s_tr, q_tr = make_dataset(Xtr, amp_tr, ph_tr, scenario, "scan",
                                  seed_off=0, add_angle=True)
        s_te, q_te = make_dataset(Xte, amp_te, ph_te, scenario, "scan",
                                  seed_off=1, add_angle=True)
        row = {}
        row["单次HRRP+CNN1D"] = cnn1d_acc(s_tr, ytr, s_te, yte, len(NAMES),
                                          epochs=100)
        esn = ESN(N_RNG + 3, n_res=512)
        row["4波束x3步+ESN+角度"] = ce_readout(
            esn.features(q_tr), ytr, esn.features(q_te), yte,
            len(NAMES), epochs=200)[0]
        res[scenario] = row
        for k, v in row.items():
            print(f"   {k}: {v:.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(SCEN)); w = 0.35
    for j, arm in enumerate(["单次HRRP+CNN1D", "4波束x3步+ESN+角度"]):
        ax.bar(x + j * w, [res[s][arm] for s in SCEN], w, label=arm)
    ax.set_xticks(x + w / 2, list(SCEN))
    ax.axhline(1 / len(NAMES), color="k", ls=":", label="随机")
    ax.set_ylabel("测试准确率"); ax.set_ylim(0, 1.05)
    ax.set_title("KITTI 道路对象: 单次 vs 扫描链")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(OUT / "acc_bars.png", dpi=130); plt.close()
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("done in %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
