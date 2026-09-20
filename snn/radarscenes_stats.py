# -*- coding: utf-8 -*-
"""RadarScenes 统计标定脚本: 真实车载毫米波雷达点云的类分布 / 多普勒 / RCS 统计.

用途: 给散射体回波仿真器标定真实统计 (杂波幅度分布、各类多普勒范围、点数分布),
并作为"真实数据分类基准"的数据源 (带 track_id 和 label_id).

数据: data/radarscenes/RadarScenes.zip 解压后
  RadarScenes/sequence_*/radar_data.h5  (结构化数组, 字段含
  x,y,z, range, azimuth, range_rate, rcs, vr, vr_compensated,
  timestamp, sensor_id, label_id, track_id)

用法:
  python radarscenes_stats.py --root ../data/radarscenes/RadarScenes [--seq 1]
输出: outputs_isal/radarscenes/ 下的统计图 + stats.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
OUT = HERE / "outputs_isal" / "radarscenes"
OUT.mkdir(parents=True, exist_ok=True)

# RadarScenes label_id (官方): 0 car, 1 large_vehicle, 2 truck, 3 bus, 4 train,
# 5 bicycle, 6 motorized_two_wheeler, 7 pedestrian, 8 pedestrian_group,
# 9 animal, 10 other, 11 unlabeled  (以数据实际为准, 运行时打印核对)
LABELS = ["car", "large_vehicle", "truck", "bus", "train", "bicycle",
          "motorized_2w", "pedestrian", "ped_group", "animal", "other"]


def load_seq(h5_path):
    import h5py
    with h5py.File(h5_path, "r") as f:
        def find(name, obj):
            if isinstance(obj, h5py.Dataset) and "radar" in name.lower():
                find.hit = (name, obj)
        find.hit = None
        f.visititems(find)
        if find.hit is None:  # 兜底: 取第一个数据集
            f.visititems(lambda n, o: setattr(find, "hit", (n, o))
                         if find.hit is None and isinstance(o, h5py.Dataset) else None)
        name, ds = find.hit
        print("  dataset:", name, "shape:", ds.shape, "dtype fields:",
              ds.dtype.names if ds.dtype.names else ds.dtype, flush=True)
        return ds[()]


def stats(d, tag):
    out = {}
    lid = d["label_id"].astype(int) if "label_id" in d.dtype.names else None
    dop = d["vr_compensated"] if "vr_compensated" in d.dtype.names else d["range_rate"]
    rcs = d["rcs"]
    if lid is not None:
        cnt = np.bincount(np.clip(lid, 0, 10), minlength=11)
        out["class_counts"] = {LABELS[i]: int(cnt[i]) for i in range(11)}
        per_cls_dop = {LABELS[i]: [float(np.mean(dop[lid == i])),
                                   float(np.std(dop[lid == i]))]
                       for i in range(11) if (lid == i).sum() > 50}
        out["doppler_mean_std_per_class"] = per_cls_dop
    out["rcs_mean_std"] = [float(rcs.mean()), float(rcs.std())]
    out["n_detections"] = int(len(dop))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if lid is not None:
        keep = [i for i in range(11) if cnt[i] > 0]
        axes[0].bar([LABELS[i] for i in keep], [cnt[i] for i in keep])
        axes[0].set(title="类分布 (%s)" % tag, ylabel="检测点数")
        axes[0].tick_params(axis="x", rotation=35, labelsize=8)
        axes[0].set_yscale("log")
        cls_shown = [i for i in range(11) if (lid == i).sum() > 200]
        for i in cls_shown:
            axes[1].hist(dop[lid == i], bins=100, histtype="step",
                         density=True, label=LABELS[i])
        axes[1].set(title="各类多普勒分布 (vr_compensated)",
                    xlabel="m/s", ylabel="密度", xlim=(-20, 20))
        axes[1].legend(fontsize=7)
    axes[2].hist(rcs, bins=100, density=True)
    axes[2].set(title="RCS 分布", xlabel="dBsm?", ylabel="密度", yscale="log")
    plt.tight_layout()
    plt.savefig(OUT / f"stats_{tag}.png", dpi=130); plt.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE / ".." / "data" / "radarscenes"
                                          / "RadarScenes"))
    ap.add_argument("--seq", type=int, default=None, help="只处理第 N 个序列")
    a = ap.parse_args()
    root = Path(a.root)
    seqs = sorted(root.glob("sequence_*"))
    if a.seq is not None:
        seqs = seqs[a.seq:a.seq + 1]
    print("sequences found:", len(seqs), flush=True)
    all_stats = {}
    for s in seqs:
        h5 = s / "radar_data.h5"
        if not h5.exists():
            continue
        print("[load]", h5, flush=True)
        d = load_seq(h5)
        all_stats[s.name] = stats(d, s.name)
        print("  ->", all_stats[s.name]["n_detections"], "detections", flush=True)
    with open(OUT / "stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print("saved -> %s" % (OUT / "stats.json"), flush=True)


if __name__ == "__main__":
    main()
