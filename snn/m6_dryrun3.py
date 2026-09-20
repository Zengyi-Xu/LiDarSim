# -*- coding: utf-8 -*-
"""M6 干跑 v3 — 数据量与学习容量: n_tr=5000, 2 层 MLP, +5dB 工作点."""
import sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config as C
from semantic_sim import SemanticEchoSim, CLASSES
from cls_ceiling import ridge_cls, onehot
from experiment import ridge_readout
from m6_dryrun2 import block, demod, feat_profile, feat_pergate, HALF

SEED = C.SEED
torch.manual_seed(SEED)


def mlp2(Ftr, ytr, Fte, yte, hidden=256, epochs=150, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Ftr - mu) / sd, (Fte - mu) / sd
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(hidden // 2, 4))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lf = torch.nn.CrossEntropyLoss()
    Ytr = ytr.long()
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = lf(net(Xtr), Ytr); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        a_tr = float((net(Xtr).argmax(1) == Ytr).float().mean())
        a_te = float((net(Xte).argmax(1) == yte.long()).float().mean())
    return a_tr, a_te


def run(snr, n_pulses, n_tr=5000, n_te=800):
    g1 = torch.Generator().manual_seed(SEED + int(snr * 10) + 7000)
    g2 = torch.Generator().manual_seed(SEED + int(snr * 10) + 8000)
    y_tr = torch.randint(0, 4, (n_tr,), generator=g1)
    y_te = torch.randint(0, 4, (n_te,), generator=g2)
    sim = SemanticEchoSim(n_pulses, C.M_BINS)
    cube_tr, sig, gate_tr = sim.make_batch(n_tr, snr, y_tr.tolist(), g1)
    cube_te, _, gate_te = sim.make_batch(n_te, snr, y_te.tolist(), g2)
    zb_tr, _ = demod(block(cube_tr, gate_tr))
    zb_te, _ = demod(block(cube_te, gate_te))
    row = {}
    row["profile+ridge"] = ridge_cls(feat_profile(zb_tr), y_tr, feat_profile(zb_te), y_te)
    row["profile+MLP2"] = mlp2(feat_profile(zb_tr), y_tr, feat_profile(zb_te), y_te)
    row["pergate+MLP2"] = mlp2(feat_pergate(zb_tr), y_tr, feat_pergate(zb_te), y_te)
    print("SNR %+.0f dB CPI=%d (n_tr=%d):" % (snr, n_pulses, n_tr), flush=True)
    for k, (a, b) in row.items():
        print("   %-16s train=%.3f test=%.3f" % (k, a, b), flush=True)
    # 混淆矩阵 (最佳臂)
    sco_te, _ = ridge_readout(feat_profile(zb_tr), onehot(y_tr),
                              feat_profile(zb_te), alpha=1.0)
    pred = sco_te.argmax(1)
    conf = torch.zeros(4, 4, dtype=torch.long)
    for t, p in zip(y_te.long(), pred):
        conf[t, p] += 1
    print("   混淆 (行=真, profile+ridge):", flush=True)
    for k in range(4):
        print("     %-10s %s" % (CLASSES[k], conf[k].tolist()), flush=True)
    return row


if __name__ == "__main__":
    t0 = time.time()
    for snr in (0.0, 5.0):
        run(snr, 64)
    run(0.0, 256)
    print("done in %.1f s" % (time.time() - t0), flush=True)
