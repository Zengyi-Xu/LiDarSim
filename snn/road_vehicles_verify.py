# -*- coding: utf-8 -*-
"""V8: 合成 ↔ 真实一致性（verify/2026-09-25）。

与 KITTI 全量同口径对比：
  - 4 类合并：car={sedan,suv}, truck={truck,bus}, cyclist={motorcycle,bicycle},
    pedestrian={pedestrian}（对应 road_kitti_experiment.MERGE 的 car/truck/pedestrian/cyclist）
  - 预算对齐：KITTI 全量 train=21,558/test=7,188 -> 合成 5390/类(75%切分) => 21,560/7,184
  - 3 种子；英文 UTF-8 键
对照目标：KITTI road_crossing scan 0.924±0.001（V6 本机实测）

输出：outputs_isal/road_vehicles_verify/results_4class.json
"""
import json, time, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "outputs_isal" / "road_vehicles_verify"
OUT.mkdir(parents=True, exist_ok=True)

import isal_range_profile as irp
from isal_range_profile import make_scatterers, ESN, ce_readout, cnn1d_acc, N_RNG, SEED
from isal_3dview import make_dataset_fast, SCEN
from road_vehicles import gen_object, to_profile_coords, resample

MERGE7to4 = {0: 0, 1: 0, 2: 1, 3: 1, 4: 3, 5: 3, 6: 2}   # 7类idx -> KITTI 4类idx
NAMES4 = ["car", "truck", "pedestrian", "cyclist"]
N_PER = 7186          # per merged class before 75% split -> train 5390, test 1796


def gen_merged(seed):
    """生成 4 合并类的合成数据集：每类 N_PER 个（按原 7 类原型均匀来源）。"""
    rng = np.random.default_rng(seed)
    src = {0: [0, 1], 1: [2, 3], 2: [6], 3: [4, 5]}
    X, y = [], []
    for c4 in range(4):
        for i in range(N_PER):
            c7 = src[c4][i % len(src[c4])]
            X.append(gen_object(c7, rng))
            y.append(c4)
    return X, np.array(y, dtype=np.int64)


def run_seed(seed):
    t0 = time.time()
    X, y = gen_merged(seed)
    n = len(X)
    rng = np.random.default_rng(seed + 500)
    tr_i, te_i = [], []
    for c in range(4):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n75 = int(len(idx) * 0.75)
        tr_i += idx[:n75].tolist()
        te_i += idx[n75:].tolist()
    Xtr = resample([to_profile_coords(X[i]) for i in tr_i])
    Xte = resample([to_profile_coords(X[i]) for i in te_i])
    ytr, yte = y[tr_i], y[te_i]
    print("[seed %d] gen+resample %.0fs train=%d test=%d" %
          (seed, time.time() - t0, len(Xtr), len(Xte)), flush=True)
    amp_tr, ph_tr = make_scatterers(Xtr, SEED + 10 * seed)
    amp_te, ph_te = make_scatterers(Xte, SEED + 10 * seed + 1)
    row = {"seed": seed, "n_train": len(Xtr), "n_test": len(Xte)}
    for scenario in SCEN:
        s_tr, q_tr = make_dataset_fast(Xtr, amp_tr, ph_tr, scenario, "scan",
                                       seed_off=2 * seed, add_angle=True)
        s_te, q_te = make_dataset_fast(Xte, amp_te, ph_te, scenario, "scan",
                                       seed_off=2 * seed + 1, add_angle=True)
        single = cnn1d_acc(s_tr, ytr, s_te, yte, 4, epochs=100, seed=seed)
        esn = ESN(N_RNG + 3, n_res=512)
        scan = ce_readout(esn.features(q_tr), ytr, esn.features(q_te), yte,
                          4, epochs=200, seed=seed)[0]
        row[scenario] = {"single_hrrp_cnn1d": single,
                         "scan_chain_esn_angle": scan}
        print("  seed=%d %s: single=%.3f scan=%.3f (%.0fs)" %
              (seed, scenario, single, scan, time.time() - t0), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    rows = [run_seed(s) for s in args.seeds]
    summary = {}
    for sc in SCEN:
        sgl = [r[sc]["single_hrrp_cnn1d"] for r in rows]
        scn = [r[sc]["scan_chain_esn_angle"] for r in rows]
        summary[sc] = {"single_mean": float(np.mean(sgl)),
                       "single_std": float(np.std(sgl)),
                       "scan_mean": float(np.mean(scn)),
                       "scan_std": float(np.std(scn))}
    out = {"task": "V8 synthetic-vs-real, 4-class merged, KITTI-parity budget",
           "n_per_merged_class": N_PER, "classes": NAMES4,
           "kitti_reference": {"road_crossing": {"single": 0.871, "scan": 0.924},
                               "uav_cap": {"single": 0.754, "scan": 0.860}},
           "seeds": args.seeds, "per_seed": rows, "summary": summary}
    with open(OUT / "results_4class.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
