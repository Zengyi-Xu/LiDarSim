# -*- coding: utf-8 -*-
"""KITTI 3D 道路目标点云提取器 -> 道路对象库 (npz), 供散射体回波仿真使用.

输入: KITTI 3D object detection 目录 (training/{velodyne,label_2,calib}).
输出: data/road_objects.npz
  points : object array, 每个 (Ni,3) float32 (velo 坐标系, 米)
  labels : int8 (0..K-1), label_names 见 ROAD_CLASSES
  meta   : frame/occ/trunc/n_points (供信息受限分析: 遮挡/稀疏度)

框在相机坐标系, 点在 velo 坐标系: 先把点变换到 rect 相机系 (Tr_velo_to_cam +
R0_rect), 再做框内裁剪; 提取完存回 velo 系.

用法:
  python road_object_extractor.py --root <KITTI>/training --out ../data/road_objects.npz
  python road_object_extractor.py --selftest          # 无数据自测 (合成小场景)
"""
import argparse, time
from pathlib import Path
import numpy as np

ROAD_CLASSES = ["car", "van", "truck", "pedestrian", "person_sitting",
                "cyclist", "tram"]
KEEP = {"car", "van", "truck", "pedestrian", "cyclist"}


def read_calib(path):
    mats = {}
    for line in open(path):
        if ":" not in line:
            continue
        key, vals = line.split(":", 1)
        mats[key.strip()] = np.fromstring(vals, sep=" ")
    return mats["Tr_velo_to_cam"].reshape(3, 4), mats["R0_rect"].reshape(3, 3)


def extract_objects(velo_bin, label_txt, calib_txt, min_points=8):
    """返回 list[dict(points(Ni,3) velo系, cls, n, occ, trunc)]"""
    pts = np.fromfile(velo_bin, dtype=np.float32).reshape(-1, 4)[:, :3]
    tr, r0 = read_calib(calib_txt)
    pr = (r0 @ (tr[:, :3] @ pts.T + tr[:, 3:4])).T          # velo -> rect cam
    objs = []
    for line in open(label_txt):
        f = line.split()
        cls = f[0].lower()
        if cls not in KEEP:
            continue
        trunc, occ = float(f[1]), int(f[2])
        h, w, l = map(float, f[8:11])
        x, y, z = map(float, f[11:14])                      # 底面中心 (cam, y向下)
        ry = float(f[14])
        center = np.array([x, y - h / 2, z])                # 底面中心 -> 几何中心
        cr, sr = np.cos(-ry), np.sin(-ry)
        R = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
        rel = (pr - center) @ R.T
        inside = ((np.abs(rel[:, 0]) <= w / 2) & (np.abs(rel[:, 1]) <= h / 2)
                  & (np.abs(rel[:, 2]) <= l / 2))
        if inside.sum() >= min_points:
            objs.append(dict(points=pts[inside].astype(np.float32), cls=cls,
                             n=int(inside.sum()), occ=occ, trunc=trunc))
    return objs


def build(root, out, min_points=8, max_frames=None):
    root, out = Path(root), Path(out)
    vdir, ldir, cdir = root / "velodyne", root / "label_2", root / "calib"
    ids = sorted(p.stem for p in vdir.glob("*.bin"))
    if max_frames:
        ids = ids[:max_frames]
    lab2idx = {c: i for i, c in enumerate(ROAD_CLASSES)}
    P, L, META = [], [], []
    t0 = time.time()
    for k, fid in enumerate(ids):
        for o in extract_objects(vdir / f"{fid}.bin", ldir / f"{fid}.txt",
                                 cdir / f"{fid}.txt", min_points):
            P.append(o["points"])
            L.append(lab2idx[o["cls"]])
            META.append((int(fid), o["occ"], o["trunc"], o["n"]))
        if k % 500 == 0:
            print(f"  frame {k}/{len(ids)}, objects={len(P)}", flush=True)
    P = np.array(P, dtype=object)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, points=P, labels=np.array(L, dtype=np.int8),
             meta=np.array(META, dtype=np.int32),
             class_names=np.array(ROAD_CLASSES))
    print("saved %d objects -> %s (%.1f s)" % (len(P), out, time.time() - t0))
    return P, np.array(L), np.array(META)


def stats_fig(P, L, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cnt = np.bincount(L, minlength=len(ROAD_CLASSES))
    axes[0].bar(ROAD_CLASSES, cnt)
    axes[0].set(title="道路对象库: 类分布", ylabel="对象数")
    axes[0].tick_params(axis="x", rotation=30)
    sizes = np.array([len(p) for p in P])
    axes[1].hist(sizes, bins=60)
    axes[1].set(title="每对象点数分布", xlabel="点数", ylabel="频数", yscale="log")
    axes[1].axvline(np.median(sizes), color="r", ls="--",
                    label="中位数=%d" % np.median(sizes))
    axes[1].legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "velodyne").mkdir(); (td / "label_2").mkdir(); (td / "calib").mkdir()
        rng = np.random.default_rng(0)
        pts = rng.uniform(-10, 10, (5000, 4)).astype(np.float32)
        pts.tofile(td / "velodyne" / "000000.bin")
        eye34 = np.hstack([np.eye(3), np.zeros((3, 1))]).ravel()
        with open(td / "calib" / "000000.txt", "w") as f:
            f.write("Tr_velo_to_cam: " + " ".join("%.6f" % v for v in eye34) + "\n")
            f.write("R0_rect: " + " ".join("%.6f" % v for v in np.eye(3).ravel()) + "\n")
        # 一个 car 框: h=1.5 w=1.6 l=4.0, 底面中心 (0, 0.75, 5), ry=0
        with open(td / "label_2" / "000000.txt", "w") as f:
            f.write("Car 0.0 0 0.0 0 0 0 0 1.5 1.6 4.0 0.0 0.75 5.0 0.0\n")
        objs = extract_objects(td / "velodyne" / "000000.bin",
                               td / "label_2" / "000000.txt",
                               td / "calib" / "000000.txt", min_points=1)
        assert len(objs) == 1 and objs[0]["cls"] == "car", objs
        # 理论点数: 均匀密度下框体积占比
        expect = 5000 * (1.5 * 1.6 * 4.0) / 8000
        got = objs[0]["n"]
        ok = abs(got - expect) / expect < 0.25
        print("selftest: extracted %d pts (expect ~%d) %s" % (got, expect,
              "OK" if ok else "FAIL"))
        assert ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--out", default=str(Path(__file__).parent / ".." / "data"
                                         / "road_objects.npz"))
    ap.add_argument("--min_points", type=int, default=8)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.root:
        P, L, M = build(a.root, a.out, a.min_points, a.max_frames)
        stats_fig(P, L, str(Path(a.out).with_suffix(".stats.png")))
    else:
        ap.print_help()
