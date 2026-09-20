# -*- coding: utf-8 -*-
"""cls_ceiling 的补充: (1) Hann 加窗 log 谱 (检验矩形窗泄漏是否压制近载波边带);
(2) acorr+ridge 逐类准确率; (3) 256 脉冲切成 4 个 CPI 多数投票 = 航迹级融合模拟."""
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config as C
from cls_ceiling import (MicroDopplerSim, feat_acorr, ridge_cls, onehot, SEED, CLASSES)

def feat_logspec_hann(z):
    N = z.shape[-1]
    w = torch.hann_window(N)
    Z = torch.fft.fftshift(torch.fft.fft(z * w, dim=-1), -1)
    return torch.log(Z.abs() ** 2 + 1e-6)

def run_detail(snr_db, n_pulses=64, n_tr=1600, n_te=500):
    g1 = torch.Generator().manual_seed(SEED + int(snr_db * 10) + 5000)
    g2 = torch.Generator().manual_seed(SEED + int(snr_db * 10) + 6000)
    y_tr = torch.randint(0, 4, (n_tr,), generator=g1)
    y_te = torch.randint(0, 4, (n_te,), generator=g2)
    sim = MicroDopplerSim(n_pulses, C.M_BINS)
    cube_tr, sig, gate_tr, _ = sim.make_batch(n_tr, snr_db, y_tr.tolist(), g1)
    cube_te, _, gate_te, _ = sim.make_batch(n_te, snr_db, y_te.tolist(), g2)
    idx = torch.arange(n_tr); idx_te = torch.arange(n_te)
    z_tr = cube_tr[idx, gate_tr] / sig
    z_te = cube_te[idx_te, gate_te] / sig
    # (1) Hann logspec
    a_hann = ridge_cls(feat_logspec_hann(z_tr), y_tr, feat_logspec_hann(z_te), y_te)
    # (2) acorr+ridge 逐类
    Ftr, Fte = feat_acorr(z_tr), feat_acorr(z_te)
    from experiment import ridge_readout
    from cls_ceiling import onehot
    sco_te, _ = ridge_readout(Ftr, onehot(y_tr), Fte, alpha=1.0)
    pred = sco_te.argmax(1)
    per_class = [(pred[y_te == k] == k).float().mean().item() for k in range(4)]
    return a_hann, per_class, (z_tr, z_te, y_tr, y_te, sig, gate_tr, gate_te)

def track_vote(snr_db, n_pulses=256, cpi=64, n_tr=1600, n_te=500):
    """长 CPI 切成多个子 CPI, 各自 acorr+ridge, 多数投票 (航迹级融合)"""
    g1 = torch.Generator().manual_seed(SEED + int(snr_db * 10) + 5000)
    g2 = torch.Generator().manual_seed(SEED + int(snr_db * 10) + 6000)
    y_tr = torch.randint(0, 4, (n_tr,), generator=g1)
    y_te = torch.randint(0, 4, (n_te,), generator=g2)
    sim = MicroDopplerSim(n_pulses, C.M_BINS)
    cube_tr, sig, gate_tr, _ = sim.make_batch(n_tr, snr_db, y_tr.tolist(), g1)
    cube_te, _, gate_te, _ = sim.make_batch(n_te, snr_db, y_te.tolist(), g2)
    idx = torch.arange(n_tr); idx_te = torch.arange(n_te)
    n_sub = n_pulses // cpi
    votes_tr, votes_te = [], []
    for s in range(n_sub):
        sl = slice(s * cpi, (s + 1) * cpi)
        z_tr = cube_tr[idx, gate_tr][:, sl] / sig
        z_te = cube_te[idx_te, gate_te][:, sl] / sig
        sco_te, sco_tr = __import__('experiment').ridge_readout(
            feat_acorr(z_tr), onehot(y_tr), feat_acorr(z_te), alpha=1.0)
        votes_tr.append(sco_tr.argmax(1)); votes_te.append(sco_te.argmax(1))
    Vtr = torch.stack(votes_tr, 1); Vte = torch.stack(votes_te, 1)
    def mv(V):
        oh = torch.nn.functional.one_hot(V, 4).sum(1).float()
        return oh.argmax(1)
    a_tr = float((mv(Vtr) == y_tr.long()).float().mean())
    a_te = float((mv(Vte) == y_te.long()).float().mean())
    return a_tr, a_te

for snr in (0.0, 5.0):
    a_hann, pc, _ = run_detail(snr)
    print("SNR %+.0f dB CPI=64:  hann-logspec+ridge test=%.3f | acorr 逐类=%s" %
          (snr, a_hann[1], ["%.2f" % x for x in pc]), flush=True)
print("\n航迹级融合 (256 脉冲 = 4x64 CPI 多数投票):", flush=True)
for snr in (0.0, 5.0, -5.0):
    a = track_vote(snr)
    print("  SNR %+.0f dB: 单CPI基线 acorr+ridge vs 投票 test=%.3f (train=%.3f)" % (snr, a[1], a[0]), flush=True)
