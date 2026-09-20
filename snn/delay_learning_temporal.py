# -*- coding: utf-8 -*-
"""
Temporal-pattern delay learning: a task where timing IS the information.

Classes differ ONLY in inter-spike intervals (rhythms), all with equal energy
and equal mean time. A delay tap is matched when its delay equals a class's
characteristic interval -> delay learning should find those delays and beat
fixed random delays.

This is the counterpoint to the point-cloud result (where delays gave ~0 gain):
delay learning pays off on genuinely temporal tasks, not on spatial ones.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from delay_learning_demo import train_model

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs_classify")
os.makedirs(OUT, exist_ok=True)


class ProductDelayModel(nn.Module):
    """Coincidence / autocorrelation taps: feature_j = mean_t tanh(g * x(t) * x(t-tau_j)).
    Delay-sensitive by construction: the product peaks only when tau_j matches an
    inter-spike interval. Physically = nonlinear optical sampling of autocorrelation."""

    def __init__(self, n_ch, T, n_taps, n_classes, gain=6.0,
                 learn_delays=True, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.T = T
        self.gain = gain
        tau0 = torch.rand(n_taps, generator=g) * (T - 1)
        self.tau = nn.Parameter(tau0, requires_grad=learn_delays)
        self.readout = nn.Linear(n_taps, n_classes)

    def forward(self, x):
        # x: (B, 1, T)
        B, C, T = x.shape
        tgrid = torch.arange(T, dtype=torch.float32, device=x.device)
        tau = self.tau.clamp(0, T - 1.001)               # (n_taps,)
        tj = tgrid[None, :] - tau[:, None]               # (n_taps, T)
        t0 = torch.floor(tj).long().clamp(0, T - 1)
        t1 = (t0 + 1).clamp(0, T - 1)
        frac = (tj - t0.float()).clamp(0, 1)
        x_now = x[:, 0, :]                               # (B, T)
        x0 = x_now[:, t0]                                # (B, n_taps, T)
        x1 = x_now[:, t1]
        x_del = x0 * (1 - frac) + x1 * frac
        prod = x_now[:, None, :] * x_del                 # (B, n_taps, T)
        F = torch.tanh(self.gain * prod).mean(dim=2)     # (B, n_taps)
        return self.readout(F)


def make_temporal_dataset(n_samples=1500, T=128, jitter=1.5, seed=0):
    """Six rhythm classes; 2 spikes each, equal energy, equal mean time.
    Intervals: 8, 14, 22, 32, 44, 58 steps (spanning most of the window).
    Mean time centered at T/2 for all -> timing is the only feature."""
    rng = np.random.default_rng(seed)
    intervals = [8, 14, 22, 32, 44, 58]
    n_classes = len(intervals)
    X = np.zeros((n_samples, 1, T), dtype=np.float32)
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        cls = i % n_classes
        y[i] = cls
        iv = intervals[cls]
        center = T // 2 + rng.integers(-4, 5)
        t1 = center - iv // 2 + rng.normal(0, jitter)
        t2 = center + iv // 2 + rng.normal(0, jitter)
        for tt in (t1, t2):
            tt = int(round(tt))
            if 0 <= tt < T:
                X[i, 0, tt] = 1.0
    return torch.tensor(X), torch.tensor(y, dtype=torch.long), len(intervals)


def run_temporal(seed=0, epochs=400, n_taps=8):
    print("\n[temporal rhythm task: 6 interval classes, only %d coincidence taps]" % n_taps)
    X, y, n_classes = make_temporal_dataset(seed=seed)
    n = len(X)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    ntr = int(0.7 * n)
    tr, te = idx[:ntr], idx[ntr:]
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]
    T = X.shape[2]

    configs = {
        "A_fixed_random": dict(learn_delays=False),
        "B_delay_learn": dict(learn_delays=True),
    }
    results, tau_hist = {}, {}
    for name, cfg in configs.items():
        model = ProductDelayModel(1, T, n_taps, n_classes, gain=6.0, seed=seed, **cfg)
        t0 = model.tau.detach().numpy().copy()
        acc, _ = train_model(model, Xtr, ytr, Xte, yte, epochs=epochs, seed=seed)
        results[name] = acc
        tau_hist[name] = (t0, model.tau.detach().numpy())
        print("  %-16s acc=%.3f" % (name, acc))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    names = list(results.keys())
    accs = [results[n] for n in names]
    ax.bar(names, accs, color=["#999999", "#2a9d8f"])
    for i, v in enumerate(accs):
        ax.text(i, v + 0.01, "%.2f" % v, ha="center")
    ax.axhline(1 / 6, color="k", ls="--", alpha=0.4)
    ax.set_ylabel("test accuracy")
    ax.set_title("coincidence taps: delay learning vs fixed (6 intervals, 8 taps)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    ax = axes[1]
    t0, t1 = tau_hist["B_delay_learn"]
    ax.scatter(t0, t1, s=20, alpha=0.6, color="#2a9d8f")
    for iv in (8, 14, 22, 32, 44, 58):
        ax.axhline(iv, color="r", ls=":", alpha=0.4)
    ax.plot([0, T - 1], [0, T - 1], "k--", alpha=0.4)
    ax.set_xlabel("initial delay (steps)")
    ax.set_ylabel("learned delay (steps)")
    ax.set_title("learned delays cluster near class intervals")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "delay_learning_temporal.png"), dpi=150)
    plt.close(fig)
    print("  saved delay_learning_temporal.png")
    return results


def main():
    t0 = time.time()
    res = run_temporal()
    np.savez(os.path.join(OUT, "delay_learning_temporal.npz"),
             **{k: v for k, v in res.items()})
    print("\nfinished in %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
