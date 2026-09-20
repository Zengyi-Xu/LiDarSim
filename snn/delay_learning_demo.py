# -*- coding: utf-8 -*-
"""
Delay-learning demo: can TRAINABLE delays beat a fixed random reservoir?

This directly tests the core innovation claim of the postdoc plan:
a TFLN electro-optic programmable delay element = a learnable synaptic delay.

Model (digital twin of the photonic tap):
  feature_j = mean_t tanh( gain * sum_c W_jc * x_c(t - tau_j) )
where tau_j is a CONTINUOUS learnable delay implemented by linear interpolation
along the time axis (differentiable).

Compare on ModelNet40 10-class subset:
  A) fixed random delays, train readout only        (= current reservoir)
  B) learnable delays, train readout + delays        (= delay learning)
  C) learnable delays + learnable input weights W    (= full)

Also report the learned delay distribution vs the initial one.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classification_demo import load_modelnet_subset, encode_coords

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs_classify")
os.makedirs(OUT, exist_ok=True)

DEVICE = "cpu"


class DelayTapModel(nn.Module):
    """n_taps delay taps over C input channels; continuous learnable delays."""

    def __init__(self, n_ch, T, n_taps, n_classes, gain=4.0,
                 learn_delays=True, learn_W=False, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.T = T
        self.gain = gain
        self.n_ch = n_ch
        # learnable delays in units of time steps
        tau0 = torch.rand(n_taps, generator=g) * (T - 1)
        self.tau = nn.Parameter(tau0, requires_grad=learn_delays)
        # fixed or learnable input mixing W (n_taps, n_ch)
        W0 = torch.randn(n_taps, n_ch, generator=g) / np.sqrt(n_ch)
        self.W = nn.Parameter(W0, requires_grad=learn_W)
        self.readout = nn.Linear(n_taps, n_classes)

    def forward(self, x):
        # x: (B, C, T) continuous rates in [0,1]
        B, C, T = x.shape
        tgrid = torch.arange(T, dtype=torch.float32, device=x.device)
        tau = self.tau.clamp(0, T - 1.001)               # (n_taps,)
        tj = tgrid[None, :] - tau[:, None]               # (n_taps, T)
        t0 = torch.floor(tj).long().clamp(0, T - 1)      # (n_taps, T)
        t1 = (t0 + 1).clamp(0, T - 1)
        frac = (tj - t0.float()).clamp(0, 1)             # (n_taps, T)
        x0 = x[:, :, t0]                                 # (B, C, n_taps, T)
        x1 = x[:, :, t1]
        x_shift = x0 * (1 - frac) + x1 * frac            # (B, C, n_taps, T)
        u = torch.einsum("bcjt,jc->bjt", x_shift, self.W)   # (B, n_taps, T)
        F = torch.tanh(self.gain * u).mean(dim=2)        # (B, n_taps)
        return self.readout(F)


def train_model(model, Xtr, ytr, Xte, yte, epochs=300, lr=3e-3, batch=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xtr[idx].to(DEVICE)
            yb = ytr[idx].to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte.to(DEVICE)).argmax(dim=1).cpu().numpy()
    return (pred == yte.numpy()).mean(), pred


def run_delay_learning(n_classes=10, n_train_per=150, n_test_per=40,
                       n_points=256, n_taps=64, seed=0, epochs=400):
    print("\n[delay learning on ModelNet40: %d classes, %d taps]" % (n_classes, n_taps))
    (Xtr_all, ytr_all), (Xte_all, yte_all), classes = load_modelnet_subset(
        n_classes, n_points, seed)
    rng = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    for c in classes:
        idx_tr = np.where(ytr_all == c)[0]
        idx_te = np.where(yte_all == c)[0]
        tr_idx.append(rng.choice(idx_tr, min(n_train_per, len(idx_tr)), replace=False))
        te_idx.append(rng.choice(idx_te, min(n_test_per, len(idx_te)), replace=False))
    tr_idx = np.concatenate(tr_idx)
    te_idx = np.concatenate(te_idx)
    Xtr, ytr = Xtr_all[tr_idx], ytr_all[tr_idx]
    Xte, yte = Xte_all[te_idx], yte_all[te_idx]
    label_map = {c: i for i, c in enumerate(classes)}
    ytr = np.vectorize(label_map.get)(ytr)
    yte = np.vectorize(label_map.get)(yte)

    # continuous rates (not Bernoulli) for differentiability: use coord encoding
    def encode_cont(X):
        B, N, _ = X.shape
        ch = np.stack([np.clip(X[:, :, 0], 0, None), np.clip(-X[:, :, 0], 0, None),
                       np.clip(X[:, :, 1], 0, None), np.clip(-X[:, :, 1], 0, None),
                       np.clip(X[:, :, 2], 0, None), np.clip(-X[:, :, 2], 0, None)],
                      axis=2)                                    # (B, N, 6)
        return torch.tensor(np.transpose(ch, (0, 2, 1)), dtype=torch.float32)

    Xtr_t = encode_cont(Xtr)
    Xte_t = encode_cont(Xte)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    yte_t = torch.tensor(yte, dtype=torch.long)
    T = Xtr_t.shape[2]

    results = {}
    tau_hist = {}
    configs = {
        "A_fixed_random": dict(learn_delays=False, learn_W=False),
        "B_delay_learn": dict(learn_delays=True, learn_W=False),
        "C_full_learn": dict(learn_delays=True, learn_W=True),
    }
    for name, cfg in configs.items():
        model = DelayTapModel(6, T, n_taps, n_classes, seed=seed, **cfg).to(DEVICE)
        tau_init = model.tau.detach().cpu().numpy().copy()
        acc, pred = train_model(model, Xtr_t, ytr_t, Xte_t, yte_t,
                                epochs=epochs, seed=seed)
        results[name] = acc
        tau_hist[name] = (tau_init, model.tau.detach().cpu().numpy())
        print("  %-16s acc=%.3f" % (name, acc))

    # figure: accuracy bar + learned delay distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    names = list(results.keys())
    accs = [results[n] for n in names]
    ax.bar(names, accs, color=["#999999", "#2a9d8f", "#e76f51"])
    for i, v in enumerate(accs):
        ax.text(i, v + 0.01, "%.2f" % v, ha="center")
    ax.axhline(0.1, color="k", ls="--", alpha=0.4)
    ax.set_ylabel("test accuracy")
    ax.set_title("delay learning vs fixed random reservoir")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    ax = axes[1]
    for name in ["B_delay_learn", "C_full_learn"]:
        t0, t1 = tau_hist[name]
        ax.scatter(t0, t1, s=14, alpha=0.6, label=name)
    lim = [0, T - 1]
    ax.plot(lim, lim, "k--", alpha=0.4, label="no change")
    ax.set_xlabel("initial delay (time steps)")
    ax.set_ylabel("learned delay (time steps)")
    ax.set_title("learned delays vs initial")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "delay_learning.png"), dpi=150)
    plt.close(fig)
    print("  saved delay_learning.png")

    return {"results": results, "n_taps": n_taps,
            "n_train": len(Xtr), "n_test": len(Xte), "n_classes": n_classes}


def main():
    t0 = time.time()
    res = run_delay_learning()
    np.savez(os.path.join(OUT, "delay_learning_results.npz"),
             acc_fixed=res["results"]["A_fixed_random"],
             acc_delay=res["results"]["B_delay_learn"],
             acc_full=res["results"]["C_full_learn"],
             n_taps=res["n_taps"], n_classes=res["n_classes"])
    print("\nfinished in %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
