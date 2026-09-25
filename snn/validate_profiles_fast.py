# -*- coding: utf-8 -*-
"""校验 make_dataset_fast（torch 批量版）与 make_dataset（numpy 原版）的一致性。

1) 同 200 个对象两条路径的输出幅值对比（期望高度相关、差异 <1e-2 量级）
2) 小规模端到端分类对比（期望准确率在统计噪声内一致）
"""
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from isal_range_profile import make_scatterers, ESN, ce_readout, cnn1d_acc, N_RNG, SEED
from isal_3dview import make_dataset, make_dataset_fast, SCEN
from road_kitti_experiment import load_library, split, NAMES
from road_vehicles import resample, to_profile_coords

P, y = load_library()
tr_i, te_i = split(P, y, n_tr_per=300)
rng = np.random.default_rng(7)
sub_te = rng.choice(te_i, 200, replace=False)
X = resample([to_profile_coords(P[i] - P[i].mean(0)) for i in sub_te])
amp, ph0 = make_scatterers(X, SEED + 1)

print("=== 1) 输出对比（200 对象, road_crossing）===")
t0 = time.time()
s_ref, q_ref = make_dataset(X, amp, ph0, "road_crossing", "scan",
                            seed_off=1, add_angle=True)
t_ref = time.time() - t0
t0 = time.time()
s_new, q_new = make_dataset_fast(X, amp, ph0, "road_crossing", "scan",
                                 seed_off=1, add_angle=True)
t_new = time.time() - t0
d_s = np.abs(s_ref - s_new).max()
d_q = np.abs(q_ref - q_new).max()
corr = np.corrcoef(q_ref.ravel(), q_new.ravel())[0, 1]
print("原版 %.1f s, 批量版 %.1f s (CPU)" % (t_ref, t_new))
print("max|Δsingle|=%.2e, max|Δscan|=%.2e, corr=%.6f" % (d_s, d_q, corr))

print("=== 2) 小规模端到端（train 300/类, test 800）===")
sub_te2 = rng.choice(te_i, 800, replace=False)
Xtr = resample([to_profile_coords(P[i] - P[i].mean(0)) for i in tr_i])
Xte = resample([to_profile_coords(P[i] - P[i].mean(0)) for i in sub_te2])
ytr, yte = y[tr_i], y[sub_te2]
amp_tr, ph_tr = make_scatterers(Xtr, SEED)
amp_te, ph_te = make_scatterers(Xte, SEED + 1)
for tag, mk in [("原版", make_dataset), ("批量版", make_dataset_fast)]:
    s_tr, q_tr = mk(Xtr, amp_tr, ph_tr, "road_crossing", "scan",
                    seed_off=0, add_angle=True)
    s_te, q_te = mk(Xte, amp_te, ph_te, "road_crossing", "scan",
                    seed_off=1, add_angle=True)
    a1 = cnn1d_acc(s_tr, ytr, s_te, yte, len(NAMES), epochs=40)
    esn = ESN(N_RNG + 3, n_res=512)
    a2 = ce_readout(esn.features(q_tr), ytr, esn.features(q_te), yte,
                    len(NAMES), epochs=80)[0]
    print("%s: CNN1D %.3f, ESN扫描链 %.3f" % (tag, a1, a2))
