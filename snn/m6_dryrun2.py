# -*- coding: utf-8 -*-
"""M6 干跑 v2 — 块处理版: 距离-慢时间块 (9 门) + 整体多普勒估计/补偿 + 块级特征.

v1 (m6_dryrun.py) 的教训: 多散射体目标的信息分布在多个距离门,
单质心门选通把腿/旋翼/翅膀的能量全丢了 -> 全员随机水平.
真实系统做法: 检测后取目标 RD 邻域块, 先补偿质心多普勒 (平移不变性
由估计+对消实现, 而非特征构造), 再做块级谱特征.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config as C
from semantic_sim import SemanticEchoSim, CLASSES
from cls_ceiling import feat_acorr, ridge_cls, mlp_cls, onehot
from experiment import ridge_readout

SEED = C.SEED
torch.manual_seed(SEED)
HALF = 4  # 块半宽: 9 门


def make(snr_db, n_pulses, n_tr=1600, n_te=500):
    g1 = torch.Generator().manual_seed(SEED + int(snr_db * 10) + 7000)
    g2 = torch.Generator().manual_seed(SEED + int(snr_db * 10) + 8000)
    y_tr = torch.randint(0, 4, (n_tr,), generator=g1)
    y_te = torch.randint(0, 4, (n_te,), generator=g2)
    sim = SemanticEchoSim(n_pulses, C.M_BINS)
    cube_tr, sig, gate_tr = sim.make_batch(n_tr, snr_db, y_tr.tolist(), g1)
    cube_te, _, gate_te = sim.make_batch(n_te, snr_db, y_te.tolist(), g2)
    return cube_tr, cube_te, sig, gate_tr, gate_te, y_tr, y_te


def block(cube, gate):
    """(B,M,N) -> (B, 2*HALF+1, N) 围绕门的块 (gate 已远离边缘)"""
    B, M, N = cube.shape
    idx = torch.arange(B)
    offs = torch.arange(-HALF, HALF + 1)
    return cube[idx[:, None], gate[:, None] + offs[None, :], :]


def demod(zblk):
    """估计整体多普勒 (块内非相干积累谱峰值) 并对消. 返回 (块, 补偿频率)"""
    Z = torch.fft.fftshift(torch.fft.fft(zblk, dim=-1), -1)
    P = Z.abs().pow(2).sum(1)                      # (B, N) 块多普勒剖面
    f_hat = (P.argmax(-1).float() / zblk.shape[-1]) - 0.5
    n = torch.arange(zblk.shape[-1])
    return zblk * torch.exp(-2j * np.pi * f_hat[:, None, None] * n), f_hat


def feat_profile(zblk_dm, n_fft=128):
    """补偿后块多普勒剖面 (对距离门非相干求和): (B, n_fft)"""
    Z = torch.fft.fftshift(torch.fft.fft(zblk_dm, n=n_fft, dim=-1), -1)
    return torch.log(Z.abs().pow(2).sum(1) + 1e-6)


def feat_pergate(zblk_dm, n_fft=128):
    """补偿后每门 log 谱展开: (B, (2*HALF+1)*n_fft)"""
    Z = torch.fft.fftshift(torch.fft.fft(zblk_dm, n=n_fft, dim=-1), -1)
    return torch.log(Z.abs().pow(2) + 1e-6).flatten(1)


def run_snr(snr, n_pulses=64):
    cube_tr, cube_te, sig, gate_tr, gate_te, y_tr, y_te = make(snr, n_pulses)
    zb_tr, f_tr = demod(block(cube_tr, gate_tr))
    zb_te, f_te = demod(block(cube_te, gate_te))
    print("SNR %+.0f dB CPI=%d: |f_hat| mean=%.4f (期望~0.125=U(0,0.25)均值,  Sanity)"
          % (snr, n_pulses, float(f_tr.abs().mean())), flush=True)
    row = {}
    row["profile+ridge"] = ridge_cls(feat_profile(zb_tr), y_tr, feat_profile(zb_te), y_te)
    row["profile+MLP"] = mlp_cls(feat_profile(zb_tr), y_tr, feat_profile(zb_te), y_te)
    row["pergate+MLP"] = mlp_cls(feat_pergate(zb_tr), y_tr, feat_pergate(zb_te), y_te)
    for k, (a, b) in row.items():
        print("   %-16s train=%.3f test=%.3f" % (k, a, b), flush=True)
    return row, (zb_tr, zb_te, y_tr, y_te)


def per_class(zb_tr, zb_te, y_tr, y_te, snr):
    sco_te, _ = ridge_readout(feat_profile(zb_tr), onehot(y_tr),
                              feat_profile(zb_te), alpha=1.0)
    pred = sco_te.argmax(1)
    pc = [(pred[y_te == k] == k).float().mean().item() for k in range(4)]
    print("   逐类 profile+ridge @%+.0f dB: %s" %
          (snr, ["%s=%.2f" % (CLASSES[k], pc[k]) for k in range(4)]), flush=True)


def vote(snr=0.0, n_pulses=256, cpi=64):
    cube_tr, cube_te, sig, gate_tr, gate_te, y_tr, y_te = make(snr, n_pulses)
    v_tr, v_te = [], []
    for s in range(0, n_pulses, cpi):
        sl = slice(s, s + cpi)
        zb_tr, _ = demod(block(cube_tr[:, :, sl], gate_tr))
        zb_te, _ = demod(block(cube_te[:, :, sl], gate_te))
        sco_te, sco_tr = ridge_readout(feat_profile(zb_tr), onehot(y_tr),
                                       feat_profile(zb_te), alpha=1.0)
        v_tr.append(sco_tr.argmax(1)); v_te.append(sco_te.argmax(1))
    def mv(V):
        oh = torch.nn.functional.one_hot(torch.stack(V, 1), 4).sum(1).float()
        return oh.argmax(1)
    a_te = float((mv(v_te) == y_te.long()).float().mean())
    print("E3 投票 %+.0f dB (%d x %d): test=%.3f" % (snr, n_pulses // cpi, cpi, a_te), flush=True)


if __name__ == "__main__":
    t0 = time.time()
    for snr in (-10.0, -5.0, 0.0):
        row, (zb_tr, zb_te, y_tr, y_te) = run_snr(snr, 64)
        if snr == 0.0:
            per_class(zb_tr, zb_te, y_tr, y_te, snr)
    print("CPI=256:", flush=True)
    run_snr(0.0, 256)
    vote(0.0)
    vote(-5.0)
    print("done in %.1f s" % (time.time() - t0), flush=True)
