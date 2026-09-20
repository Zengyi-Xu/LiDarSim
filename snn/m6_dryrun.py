# -*- coding: utf-8 -*-
"""M6 干跑: 语义目标分类真实数字 (先于笔记本结论, 保证 md 引用实测值).

E1: CPI=64, SNR {-15,-10,-5,0}, oracle 门, arms = acorr+ridge / logspec+ridge / LSM+MLP
E1b: CPI=256, SNR {-5,0}, arms += STFT+MLP (语义丰富结构需要时频特征)
E2: 能量-ML 门 (非相干距离像峰值) vs oracle 门, 门匹配率 + 门误差下分类
E3: 4x64-CPI 多数投票 = 航迹级融合 (0 dB)
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config as C
from semantic_sim import SemanticEchoSim, CLASSES
from cls_ceiling import (feat_acorr, feat_logspec, ridge_cls, mlp_cls,
                         lsm_mlp, onehot)
from experiment import ridge_readout

SEED = C.SEED
torch.manual_seed(SEED)


def feat_stft(z, n_fft=64, win=48, hop=16):
    N = z.shape[-1]
    w = torch.hann_window(win)
    frames = []
    for s in range(0, max(1, N - win + 1), hop):
        seg = z[:, s:s + win]
        if seg.shape[-1] < win:
            seg = torch.nn.functional.pad(seg, (0, win - seg.shape[-1]))
        Z = torch.fft.fftshift(torch.fft.fft(seg * w, n=n_fft), -1)
        frames.append(torch.log(Z.abs() ** 2 + 1e-6))
    return torch.cat(frames, 1)


def make(snr_db, n_pulses, n_tr=1600, n_te=500):
    g1 = torch.Generator().manual_seed(SEED + int(snr_db * 10) + 7000)
    g2 = torch.Generator().manual_seed(SEED + int(snr_db * 10) + 8000)
    y_tr = torch.randint(0, 4, (n_tr,), generator=g1)
    y_te = torch.randint(0, 4, (n_te,), generator=g2)
    sim = SemanticEchoSim(n_pulses, C.M_BINS)
    cube_tr, sig, gate_tr = sim.make_batch(n_tr, snr_db, y_tr.tolist(), g1)
    cube_te, _, gate_te = sim.make_batch(n_te, snr_db, y_te.tolist(), g2)
    idx = torch.arange(n_tr); idx_te = torch.arange(n_te)
    z_tr = cube_tr[idx, gate_tr] / sig
    z_te = cube_te[idx_te, gate_te] / sig
    return (z_tr, z_te, y_tr, y_te, sig, cube_tr, cube_te, gate_tr, gate_te)


def e1():
    res = {}
    for snr in (-15.0, -10.0, -5.0, 0.0):
        z_tr, z_te, y_tr, y_te, sig, *_ = make(snr, 64)
        row = {}
        row["acorr+ridge"] = ridge_cls(feat_acorr(z_tr), y_tr, feat_acorr(z_te), y_te)
        row["logspec+ridge"] = ridge_cls(feat_logspec(z_tr), y_tr, feat_logspec(z_te), y_te)
        row["LSM+MLP"] = lsm_mlp((z_tr, z_te), sig, y_tr, y_te)
        res[snr] = row
        print("E1 SNR %+.0f dB CPI=64:" % snr, flush=True)
        for k, (a, b) in row.items():
            print("   %-14s train=%.3f test=%.3f" % (k, a, b), flush=True)
    return res


def e1b():
    res = {}
    for snr in (-5.0, 0.0):
        z_tr, z_te, y_tr, y_te, sig, *_ = make(snr, 256)
        row = {}
        row["acorr+ridge"] = ridge_cls(feat_acorr(z_tr), y_tr, feat_acorr(z_te), y_te)
        row["STFT+MLP"] = mlp_cls(feat_stft(z_tr), y_tr, feat_stft(z_te), y_te)
        row["LSM+MLP"] = lsm_mlp((z_tr, z_te), sig, y_tr, y_te)
        res[snr] = row
        print("E1b SNR %+.0f dB CPI=256:" % snr, flush=True)
        for k, (a, b) in row.items():
            print("   %-14s train=%.3f test=%.3f" % (k, a, b), flush=True)
    return res


def e2():
    print("E2 能量-ML 门鲁棒性:", flush=True)
    for snr in (-15.0, -10.0, -5.0, 0.0):
        z_tr, z_te, y_tr, y_te, sig, cube_tr, cube_te, gate_tr, gate_te = make(snr, 64)
        ml_tr = cube_tr.abs().pow(2).sum(-1).argmax(-1)
        ml_te = cube_te.abs().pow(2).sum(-1).argmax(-1)
        m_tr = float((ml_tr == gate_tr).float().mean())
        m_te = float((ml_te == gate_te).float().mean())
        idx = torch.arange(len(y_tr)); idx_te = torch.arange(len(y_te))
        zm_tr = cube_tr[idx, ml_tr] / sig
        zm_te = cube_te[idx_te, ml_te] / sig
        acc_ml = ridge_cls(feat_acorr(zm_tr), y_tr, feat_acorr(zm_te), y_te)
        acc_or = ridge_cls(feat_acorr(z_tr), y_tr, feat_acorr(z_te), y_te)
        print("   SNR %+.0f dB: 门匹配 train/test=%.2f/%.2f | acorr+ridge oracle=%.3f ML门=%.3f"
              % (snr, m_tr, m_te, acc_or[1], acc_ml[1]), flush=True)


def e3():
    print("E3 航迹级融合 (0 dB, 4x64-CPI 投票):", flush=True)
    snr = 0.0
    g1 = torch.Generator().manual_seed(SEED + 9000)
    g2 = torch.Generator().manual_seed(SEED + 9100)
    n_tr, n_te, np_ = 1600, 500, 256
    y_tr = torch.randint(0, 4, (n_tr,), generator=g1)
    y_te = torch.randint(0, 4, (n_te,), generator=g2)
    sim = SemanticEchoSim(np_, C.M_BINS)
    cube_tr, sig, gate_tr = sim.make_batch(n_tr, snr, y_tr.tolist(), g1)
    cube_te, _, gate_te = sim.make_batch(n_te, snr, y_te.tolist(), g2)
    idx = torch.arange(n_tr); idx_te = torch.arange(n_te)
    v_tr, v_te = [], []
    for s in range(0, np_, 64):
        sl = slice(s, s + 64)
        z_tr = cube_tr[idx, gate_tr][:, sl] / sig
        z_te = cube_te[idx_te, gate_te][:, sl] / sig
        sco_te, sco_tr = ridge_readout(feat_acorr(z_tr), onehot(y_tr),
                                       feat_acorr(z_te), alpha=1.0)
        v_tr.append(sco_tr.argmax(1)); v_te.append(sco_te.argmax(1))
    def mv(V):
        oh = torch.nn.functional.one_hot(torch.stack(V, 1), 4).sum(1).float()
        return oh.argmax(1)
    a_tr = float((mv(v_tr) == y_tr.long()).float().mean())
    a_te = float((mv(v_te) == y_te.long()).float().mean())
    print("   投票 train=%.3f test=%.3f" % (a_tr, a_te), flush=True)


if __name__ == "__main__":
    t0 = time.time()
    e1(); e1b(); e2(); e3()
    print("done in %.1f s" % (time.time() - t0), flush=True)
