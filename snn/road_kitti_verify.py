# -*- coding: utf-8 -*-
"""V6/V7 复核：KITTI 道路对象分类，多种子 + 预算扫描（verify/2026-09-25）。

在 road_kitti_experiment.py 基础上扩展：
  --seed 贯穿 split / make_scatterers / seed_off / cnn1d_acc / ce_readout
  --n-tr-per 预算扫描（None = 每类 75%，全量）
  英文 UTF-8 键名（修复原 results.json 键名乱码问题），逐种子明细 + 均值±std

输出：outputs_isal/road_kitti_verify/results_<tag>.json
用法：
  python road_kitti_verify.py --seeds 0 1 2                      # V6 全量
  python road_kitti_verify.py --seeds 0 1 2 --n-tr-per 300       # V7 预算点
"""
import json, time, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
NPZ = HERE / ".." / "data" / "road_objects.npz"
OUT = HERE / "outputs_isal" / "road_kitti_verify"
OUT.mkdir(parents=True, exist_ok=True)

import isal_range_profile as irp
from isal_range_profile import make_scatterers, ESN, ce_readout, cnn1d_acc, N_RNG, SEED
from isal_3dview import make_dataset_fast, SCEN
from road_vehicles import resample, to_profile_coords

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


def split(P, y, n_tr_per, seed):
    rng = np.random.default_rng(seed)
    tr_i, te_i = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n75 = int(len(idx) * 0.75)
        n_tr = n75 if n_tr_per is None else min(n_tr_per, n75)
        tr_i += idx[:n_tr].tolist()
        te_i += idx[n_tr:].tolist()
    return tr_i, te_i


def run_one(P, y, n_tr_per, seed, chunk):
    tr_i, te_i = split(P, y, n_tr_per, seed)
    Xtr = resample([to_profile_coords(P[i] - P[i].mean(0)) for i in tr_i])
    Xte = resample([to_profile_coords(P[i] - P[i].mean(0)) for i in te_i])
    ytr, yte = y[tr_i], y[te_i]
    amp_tr, ph_tr = make_scatterers(Xtr, SEED + 10 * seed)
    amp_te, ph_te = make_scatterers(Xte, SEED + 10 * seed + 1)
    row = {"seed": seed, "n_train": len(Xtr), "n_test": len(Xte)}
    for scenario in SCEN:
        s_tr, q_tr = make_dataset_fast(Xtr, amp_tr, ph_tr, scenario, "scan",
                                       seed_off=2 * seed, add_angle=True,
                                       chunk=chunk)
        s_te, q_te = make_dataset_fast(Xte, amp_te, ph_te, scenario, "scan",
                                       seed_off=2 * seed + 1, add_angle=True,
                                       chunk=chunk)
        single = cnn1d_acc(s_tr, ytr, s_te, yte, len(NAMES), epochs=100,
                           seed=seed)
        esn = ESN(N_RNG + 3, n_res=512)
        scan = ce_readout(esn.features(q_tr), ytr, esn.features(q_te), yte,
                          len(NAMES), epochs=200, seed=seed)[0]
        row[scenario] = {"single_hrrp_cnn1d": single,
                         "scan_chain_esn_angle": scan}
        print("  seed=%d %s: single=%.3f scan=%.3f" % (seed, scenario, single,
                                                       scan), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-tr-per", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or ("full" if args.n_tr_per is None else "ntr%d" % args.n_tr_per)

    P, y = load_library()
    print("[lib] objects=%d counts=%s" % (len(P), np.bincount(y, minlength=4).tolist()),
          flush=True)
    t0 = time.time()
    rows = []
    for seed in args.seeds:
        print("[seed %d] start (n_tr_per=%s)" % (seed, args.n_tr_per), flush=True)
        rows.append(run_one(P, y, args.n_tr_per, seed, args.chunk))
    acc = {sc: {arm: [r[sc][arm] for r in rows] for arm in
                ["single_hrrp_cnn1d", "scan_chain_esn_angle"]}
           for sc in SCEN}
    summary = {}
    for sc in SCEN:
        s = {a: [r[sc][a] for r in rows] for a in
             ["single_hrrp_cnn1d", "scan_chain_esn_angle"]}
        summary[sc] = {
            "single_mean": float(np.mean(s["single_hrrp_cnn1d"])),
            "single_std": float(np.std(s["single_hrrp_cnn1d"])),
            "scan_mean": float(np.mean(s["scan_chain_esn_angle"])),
            "scan_std": float(np.std(s["scan_chain_esn_angle"])),
            "gain_mean": float(np.mean([b - a for a, b in
                                        zip(s["single_hrrp_cnn1d"],
                                            s["scan_chain_esn_angle"])])),
            "gain_std": float(np.std([b - a for a, b in
                                      zip(s["single_hrrp_cnn1d"],
                                          s["scan_chain_esn_angle"])])),
        }
    out = {"tag": tag, "n_tr_per": args.n_tr_per, "seeds": args.seeds,
           "per_seed": rows, "per_class_names": NAMES, "summary": summary,
           "wall_time_s": time.time() - t0}
    with open(OUT / ("results_%s.json" % tag), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[done] %s  wall=%.0fs" % (OUT / ("results_%s.json" % tag),
                                     time.time() - t0), flush=True)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
