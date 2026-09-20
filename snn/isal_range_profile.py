# -*- coding: utf-8 -*-
"""
M4: simulated HRRP range profiles + turntable ISAL from ModelNet40 point clouds.

Physical chain under test:
  chirp -> (on-chip dispersive grating) pulse compression -> coherent range
  profile (HRRP) -> reservoir/readout classification, and
  multi-aspect coherent profiles -> turntable ISAL 2D image.

Scatterer model (0th order, honest limitations noted):
  surface points = independent point scatterers, complex reflectivity
  (Rayleigh amplitude, uniform phase). Coherent sum per range cell.
  Speckle emerges from the coherent sum; aspect decorrelation is modelled by
  an Ornstein-Uhlenbeck random walk of each scatterer's phase with a tunable
  coherence angle theta_c (theta_c=inf -> ideal coherent point targets).

Effective wavelength is scaled to the range cell (fully-developed speckle
regime, lambda << cell); absolute lambda does not change the statistics we
study. Range window D maps to platform choice via the M3b delay budget:
  D = 4 m    -> SiN-scale window (vehicle / short-range radar framing)
  D = 0.15 m -> TFLN single-grating window (small-target imaging framing)

Experiments:
  1) example profiles + ISAL images (coherent vs coherence-limited)
  2) 10-class classification: ridge on profile / CE on profile / ESN+CE / ISAL image+CE
  3) aspect generalization: train +/-30 deg, test at growing offsets
  4a) SNR sweep
  4b) coherence-angle sweep: incoherent-sequence ESN vs coherent ISAL image
  5) small-target case D=0.15 m (TFLN window)
"""
import os
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classification_demo import load_modelnet_subset

MN40 = ["airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl",
        "car", "chair", "cone", "cup", "curtain", "desk", "door", "dresser",
        "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop",
        "mantel", "monitor", "night_stand", "person", "piano", "plant",
        "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table",
        "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs_isal")
os.makedirs(OUT, exist_ok=True)
TFLN_RESULTS = r"D:\kimi_workspace\tfln-dispersion-lab\results\snn"
os.makedirs(TFLN_RESULTS, exist_ok=True)

SEED = 0
N_RNG = 128          # range cells
N_ANG_ISAL = 256     # aspect samples for ISAL imaging
TH_ISAL = np.deg2rad(3.0)   # ISAL coherent aperture half-width
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# scatterer model
# ---------------------------------------------------------------------------
def make_scatterers(X, seed=SEED):
    """Attach complex reflectivity to each point. X: (N, P, 3) normalized."""
    rng = np.random.default_rng(seed)
    N, P, _ = X.shape
    amp = np.sqrt(rng.exponential(1.0, (N, P)))        # Rayleigh amplitude
    ph0 = rng.uniform(0, 2 * np.pi, (N, P))            # static phase
    return amp.astype(np.float32), ph0.astype(np.float32)


def ou_phase_walk(n_ang, dth, theta_c, shape, rng):
    """OU random walk in aspect angle; stationary N(0,1). (n_ang, *shape)."""
    if np.isinf(theta_c):
        return rng.standard_normal((1,) + tuple(shape)) * np.ones((n_ang,) + tuple(shape))
    a = np.exp(-dth / theta_c)
    e = rng.standard_normal((n_ang,) + tuple(shape))
    out = np.empty_like(e)
    out[0] = e[0]
    for m in range(1, n_ang):
        out[m] = a * out[m - 1] + np.sqrt(1 - a * a) * e[m]
    return out


def complex_profiles(P, amp, ph0, thetas, D, theta_c=np.inf, rng=None,
                     snr_db=None):
    """Coherent range profiles of one object at aspects `thetas`.

    P: (Pp, 3) points (unit scale); amp, ph0: (Pp,); thetas: (M,) rad.
    Returns complex profiles (M, N_RNG). Range axis = rotated x.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    cell = D / N_RNG
    lam = cell / 8.0                      # effective wavelength (speckle regime)
    x, y = P[:, 0] * D / 2, P[:, 1] * D / 2        # object spans ~D in x/y
    ct, st = np.cos(thetas)[:, None], np.sin(thetas)[:, None]
    r = ct * x[None, :] + st * y[None, :]          # (M, Pp) line-of-sight
    dth = abs(thetas[1] - thetas[0]) if len(thetas) > 1 else 1.0
    ou = ou_phase_walk(len(thetas), dth, theta_c, (len(x),), rng)
    phase = 4 * np.pi * r / lam + ph0[None, :] + np.pi * ou
    contrib = amp[None, :] * np.exp(1j * phase)
    idx = np.floor((r + D / 2) / cell).astype(int)
    idx = np.clip(idx, 0, N_RNG - 1)
    prof = np.zeros((len(thetas), N_RNG), dtype=np.complex64)
    midx = np.repeat(np.arange(len(thetas)), len(x))
    np.add.at(prof, (midx, idx.ravel()), contrib.ravel())
    if snr_db is not None:
        p_sig = np.mean(np.abs(prof) ** 2)
        p_n = p_sig / (10 ** (snr_db / 10))
        noise = np.sqrt(p_n / 2) * (rng.standard_normal(prof.shape) +
                                    1j * rng.standard_normal(prof.shape))
        prof = prof + noise
    return prof


def isal_image(P, amp, ph0, D, theta_c, seed):
    """Turntable ISAL: 2D DFT of complex profiles over the coherent aperture."""
    rng = np.random.default_rng(seed)
    thetas = np.linspace(-TH_ISAL, TH_ISAL, N_ANG_ISAL)
    prof = complex_profiles(P, amp, ph0, thetas, D, theta_c, rng)
    w = np.hanning(N_ANG_ISAL)[:, None]
    img = np.fft.fftshift(np.fft.fft(prof * w, axis=0), axes=0)
    return np.abs(img).astype(np.float32)


# ---------------------------------------------------------------------------
# readouts
# ---------------------------------------------------------------------------
def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_acc(Xtr, ytr, Xte, yte, n_classes, alpha=1.0):
    Xtr, Xte = standardize(Xtr, Xte)
    Y = np.zeros((len(ytr), n_classes)); Y[np.arange(len(ytr)), ytr] = 1
    n, d = Xtr.shape
    if d > n:                                   # dual form (M2 lesson)
        K = Xtr @ Xtr.T + alpha * np.eye(n)
        B = Xtr.T @ np.linalg.solve(K, Y)
    else:
        B = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(d), Xtr.T @ Y)
    return float((Xte @ B).argmax(1).__eq__(yte).mean())


def ce_readout(Xtr, ytr, Xte, yte, n_classes, epochs=300, lr=3e-3, seed=SEED):
    torch.manual_seed(seed)
    Xtr, Xte = standardize(Xtr, Xte)
    lin = nn.Linear(Xtr.shape[1], n_classes)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    n = len(Xtr_t)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = lossf(lin(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = lin(Xte_t).argmax(1).numpy()
    return float((pred == yte).mean()), pred


def cnn1d_acc(Xtr, ytr, Xte, yte, n_classes, epochs=300, lr=3e-3, seed=SEED):
    torch.manual_seed(seed)
    Xtr, Xte = standardize(Xtr, Xte)
    Xtr_t = torch.tensor(Xtr[:, None, :], dtype=torch.float32)
    Xte_t = torch.tensor(Xte[:, None, :], dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    yte_t = torch.tensor(yte, dtype=torch.long)
    net = nn.Sequential(
        nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        nn.Flatten(),
        nn.Linear(Xtr.shape[1] // 4 * 32, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, n_classes))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr_t)
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = lossf(net(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xte_t).argmax(1).numpy()
    return float((pred == yte_t.numpy()).mean())


def multi_look_profiles(X, amp, ph0, D, n_looks, snr_db=None, seed_off=0):
    """Average n_looks magnitude profiles (random aspects) per sample."""
    out = np.zeros((len(X), N_RNG), dtype=np.float32)
    for i in range(len(X)):
        rng = np.random.default_rng(SEED + 9000 + seed_off + i)
        prof_sum = np.zeros(N_RNG, dtype=np.float32)
        for _ in range(n_looks):
            th = np.array([rng.uniform(0, 2 * np.pi)])
            prof_sum += np.abs(complex_profiles(X[i], amp[i], ph0[i], th, D,
                                                np.inf, rng, snr_db)[0])
        out[i] = prof_sum / n_looks
    return out


class ESN(object):
    """Echo-state reservoir, batched. Input sequence (B, T, n_in)."""
    def __init__(self, n_in, n_res=512, sr=0.95, leak=0.3, seed=SEED):
        rng = np.random.default_rng(seed)
        self.Win = (rng.standard_normal((n_res, n_in)) / np.sqrt(n_in)).astype(np.float32)
        W = rng.standard_normal((n_res, n_res)).astype(np.float32)
        W[np.random.default_rng(seed + 1).random((n_res, n_res)) > 0.1] = 0
        W *= sr / max(1e-6, np.abs(np.linalg.eigvals(W)).max())
        self.W = W
        self.leak = leak

    def features(self, U):
        B = U.shape[0]
        x = np.zeros((B, self.W.shape[0]), dtype=np.float32)
        ssum = np.zeros_like(x)
        for t in range(U.shape[1]):
            u = U[:, t, :]
            x = (1 - self.leak) * x + self.leak * np.tanh(
                u @ self.Win.T + x @ self.W.T)
            ssum += x
        return np.concatenate([ssum / U.shape[1], x], axis=1)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def get_data(n_classes=10, n_train_per=150, n_test_per=40, n_points=256):
    (Xtr_all, ytr_all), (Xte_all, yte_all), classes = load_modelnet_subset(
        n_classes, n_points, SEED)
    rng = np.random.default_rng(SEED)
    tr_idx = np.concatenate([rng.choice(np.where(ytr_all == c)[0],
                                        min(n_train_per, (ytr_all == c).sum()),
                                        replace=False) for c in classes])
    te_idx = np.concatenate([rng.choice(np.where(yte_all == c)[0],
                                        min(n_test_per, (yte_all == c).sum()),
                                        replace=False) for c in classes])
    lab = {c: i for i, c in enumerate(classes)}
    X = np.concatenate([Xtr_all[tr_idx], Xte_all[te_idx]])
    y = np.vectorize(lab.get)(np.concatenate([ytr_all[tr_idx], yte_all[te_idx]]))
    ntr = len(tr_idx)
    return X[:ntr], y[:ntr], X[ntr:], y[ntr:], classes


def profiles_dataset(X, amp, ph0, D, theta_sampler, snr_db=None, seed_off=0):
    """One magnitude profile per sample. Returns (N, N_RNG) float32."""
    out = np.zeros((len(X), N_RNG), dtype=np.float32)
    for i in range(len(X)):
        rng = np.random.default_rng(SEED + 7 + seed_off + i)
        th = np.array([theta_sampler(rng)])
        out[i] = np.abs(complex_profiles(X[i], amp[i], ph0[i], th, D,
                                         np.inf, rng, snr_db)[0])
    return out


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("[load] ModelNet40 subset ...")
    Xtr, ytr, Xte, yte, classes = get_data()
    names = [MN40[int(c)] for c in classes]
    n_classes = len(classes)
    print("  train %d test %d classes: %s" % (len(Xtr), len(Xte), names))
    amp_tr, ph_tr = make_scatterers(Xtr, SEED)
    amp_te, ph_te = make_scatterers(Xte, SEED + 1)

    # ---- Fig 1: example profiles ----------------------------------------
    D = 4.0
    show = [names.index(k) for k in ["car", "airplane", "chair", "bottle"]
            if k in names]
    fig, axes = plt.subplots(len(show), 3, figsize=(11, 2.1 * len(show)),
                             sharex=True)
    for r, ci in enumerate(show):
        i = np.where(ytr == ci)[0][0]
        for c, deg in enumerate([0, 45, 90]):
            rng = np.random.default_rng(SEED + 100 + i * 10 + c)
            th = np.array([np.deg2rad(deg)])
            pr = np.abs(complex_profiles(Xtr[i], amp_tr[i], ph_tr[i], th, D,
                                         np.inf, rng)[0])
            axes[r, c].plot(np.linspace(-D / 2, D / 2, N_RNG), pr, lw=1)
            axes[r, c].set_title("%s @ %d deg" % (names[ci], deg), fontsize=9)
            axes[r, c].set_yticks([])
    axes[-1, 1].set_xlabel("range (m)")
    fig.suptitle("Simulated HRRP range profiles (D=4 m window, 3.1 cm cells)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "m4_fig_profiles.png"), dpi=130)
    plt.close(fig)
    print("[fig] profiles  (%.0f s)" % (time.time() - t0))

    # ---- Fig 2: ISAL images, coherent vs coherence-limited ---------------
    picks = [names.index(k) for k in ["car", "airplane"] if k in names]
    tcs = [np.inf, np.deg2rad(0.3), np.deg2rad(0.05)]
    fig, axes = plt.subplots(len(picks), len(tcs), figsize=(11, 3.4 * len(picks)))
    dy = (D / N_RNG / 8) / (4 * TH_ISAL)     # cross-range pixel size (m)
    yext = N_ANG_ISAL * dy / 2
    for r, ci in enumerate(picks):
        i = np.where(ytr == ci)[0][0]
        for c, tc in enumerate(tcs):
            img = isal_image(Xtr[i], amp_tr[i], ph_tr[i], D, tc,
                             SEED + 200 + i * 10 + c)
            img_db = 20 * np.log10(img / img.max() + 1e-3)
            axes[r, c].imshow(img_db, aspect="auto", cmap="inferno",
                              extent=[-D / 2, D / 2, -yext, yext], vmin=-30)
            tcs_str = "inf" if np.isinf(tc) else "%.2f deg" % np.rad2deg(tc)
            axes[r, c].set_title("%s, theta_c=%s" % (names[ci], tcs_str),
                                 fontsize=9)
            axes[r, c].set_xlabel("range (m)")
            axes[r, c].set_ylabel("cross-range (m)")
    fig.suptitle("Turntable ISAL (aperture +/-3 deg): coherent vs coherence-limited")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "m4_fig_isal.png"), dpi=130)
    plt.close(fig)
    print("[fig] ISAL images  (%.0f s)" % (time.time() - t0))

    # ---- Exp 2: 10-class classification ----------------------------------
    print("[exp2] classification ...")
    sampler = lambda rng: rng.uniform(0, 2 * np.pi)
    Ptr = profiles_dataset(Xtr, amp_tr, ph_tr, D, sampler)
    Pte = profiles_dataset(Xte, amp_te, ph_te, D, sampler)

    acc = {}
    acc["profile_ridge"] = ridge_acc(Ptr, ytr, Pte, yte, n_classes)
    acc["profile_ce"], _ = ce_readout(Ptr, ytr, Pte, yte, n_classes)

    esn = ESN(1, n_res=512)
    acc["profile_esn_ce"], pred_esn = ce_readout(
        esn.features(Ptr[:, :, None]), ytr,
        esn.features(Pte[:, :, None]), yte, n_classes)
    print("  profile methods done (%.0f s)" % (time.time() - t0))

    Itr = np.stack([isal_image(Xtr[i], amp_tr[i], ph_tr[i], D, np.inf,
                               SEED + 300 + i) for i in range(len(Xtr))])
    Ite = np.stack([isal_image(Xte[i], amp_te[i], ph_te[i], D, np.inf,
                               SEED + 400 + i) for i in range(len(Xte))])

    def img_feat(I):
        B = I.reshape(I.shape[0], 8, 32, 8, 16).mean(axis=(1, 3))
        return B.reshape(I.shape[0], -1)                     # 32x64 -> 2048
    acc["isal_img_ce"], _ = ce_readout(img_feat(Itr), ytr, img_feat(Ite),
                                       yte, n_classes)
    print("  ISAL-image method done (%.0f s)" % (time.time() - t0))
    for k, v in acc.items():
        print("    %-16s %.3f" % (k, v))

    # ---- Exp 2b: fair baselines (1D CNN + multi-look averaging) -----------
    print("[exp2b] baselines ...")
    acc["profile_cnn"] = cnn1d_acc(Ptr, ytr, Pte, yte, n_classes)
    ML4tr = multi_look_profiles(Xtr, amp_tr, ph_tr, D, 4, seed_off=10000)
    ML4te = multi_look_profiles(Xte, amp_te, ph_te, D, 4, seed_off=11000)
    acc["4look_cnn"] = cnn1d_acc(ML4tr, ytr, ML4te, yte, n_classes)
    ML16tr = multi_look_profiles(Xtr, amp_tr, ph_tr, D, 16, seed_off=12000)
    ML16te = multi_look_profiles(Xte, amp_te, ph_te, D, 16, seed_off=13000)
    acc["16look_cnn"] = cnn1d_acc(ML16tr, ytr, ML16te, yte, n_classes)
    for k in ["profile_cnn", "4look_cnn", "16look_cnn"]:
        print("    %-16s %.3f" % (k, acc[k]))

    # confusion matrix of ESN method
    cm = np.zeros((n_classes, n_classes), int)
    for t, p in zip(yte, pred_esn):
        cm[t, p] += 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(range(len(acc)), list(acc.values()),
                tick_label=list(acc.keys()))
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("10-class ModelNet40 from one range profile")
    for i, v in enumerate(acc.values()):
        axes[0].text(i, v + 0.02, "%.2f" % v, ha="center", fontsize=8)
    axes[0].tick_params(axis="x", rotation=20)
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks(range(n_classes), names, rotation=45, ha="right",
                       fontsize=8)
    axes[1].set_yticks(range(n_classes), names, fontsize=8)
    axes[1].set_title("confusion: profile + ESN + CE")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "m4_fig_classification.png"), dpi=130)
    plt.close(fig)

    # ---- Exp 3: aspect generalization ------------------------------------
    print("[exp3] aspect generalization ...")
    Ptr30 = profiles_dataset(Xtr, amp_tr, ph_tr, D,
                             lambda rng: rng.uniform(-np.pi / 6, np.pi / 6),
                             seed_off=1000)
    esn3 = ESN(1, n_res=512)
    Ftr30 = esn3.features(Ptr30[:, :, None])
    offsets = [0, 30, 60, 90, 120, 150, 180]
    acc_vs_off = []
    for k, off in enumerate(offsets):
        Pte_k = profiles_dataset(
            Xte, amp_te, ph_te, D,
            lambda rng, o=off: np.deg2rad(o) + rng.uniform(-np.pi / 12,
                                                           np.pi / 12),
            seed_off=2000 + k)
        a, _ = ce_readout(Ftr30, ytr, esn3.features(Pte_k[:, :, None]),
                          yte, n_classes)
        acc_vs_off.append(a)
        print("    offset %3d deg: %.3f" % (off, a))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(offsets, acc_vs_off, "o-")
    ax.set_xlabel("aspect offset between train and test (deg)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Aspect generalization (train within +/-30 deg)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "m4_fig_aspect.png"), dpi=130)
    plt.close(fig)

    # ---- Exp 4a: SNR sweep ------------------------------------------------
    print("[exp4a] SNR sweep ...")
    snrs = [-10, 0, 10, 20, 30]
    acc_snr = []
    esn4 = ESN(1, n_res=512)
    for k, s in enumerate(snrs):
        Ptr_s = profiles_dataset(Xtr, amp_tr, ph_tr, D, sampler, snr_db=s,
                                 seed_off=3000 + k)
        Pte_s = profiles_dataset(Xte, amp_te, ph_te, D, sampler, snr_db=s,
                                 seed_off=4000 + k)
        a, _ = ce_readout(esn4.features(Ptr_s[:, :, None]), ytr,
                          esn4.features(Pte_s[:, :, None]), yte, n_classes)
        acc_snr.append(a)
        print("    SNR %3d dB: %.3f" % (s, a))

    # ---- Exp 4b: coherence-angle sweep (multi-aspect sequences) ----------
    print("[exp4b] coherence-angle sweep ...")
    tcs_sweep = [np.inf, np.deg2rad(1.0), np.deg2rad(0.3), np.deg2rad(0.1),
                 np.deg2rad(0.03)]
    n_seq = 16
    acc_tc_esn, acc_tc_img = [], []
    esn_seq = ESN(N_RNG, n_res=512)
    for k, tc in enumerate(tcs_sweep):
        def seq_profiles(X, amp, ph, off):
            out = np.zeros((len(X), n_seq, N_RNG), dtype=np.float32)
            for i in range(len(X)):
                rng = np.random.default_rng(SEED + 5000 + off + i)
                c0 = rng.uniform(0, 2 * np.pi)
                ths = c0 + np.linspace(-TH_ISAL, TH_ISAL, n_seq)
                out[i] = np.abs(complex_profiles(X[i], amp[i], ph[i], ths, D,
                                                 tc, rng))
            return out
        Str = seq_profiles(Xtr, amp_tr, ph_tr, k * 1000)
        Ste = seq_profiles(Xte, amp_te, ph_te, k * 1000 + 100)
        a, _ = ce_readout(esn_seq.features(Str), ytr, esn_seq.features(Ste),
                          yte, n_classes)
        acc_tc_esn.append(a)
        Itr_tc = np.stack([isal_image(Xtr[i], amp_tr[i], ph_tr[i], D, tc,
                                      SEED + 6000 + k * 1000 + i)
                           for i in range(len(Xtr))])
        Ite_tc = np.stack([isal_image(Xte[i], amp_te[i], ph_te[i], D, tc,
                                      SEED + 7000 + k * 1000 + i)
                           for i in range(len(Xte))])
        b, _ = ce_readout(img_feat(Itr_tc), ytr, img_feat(Ite_tc), yte,
                          n_classes)
        acc_tc_img.append(b)
        tcs_str = "inf" if np.isinf(tc) else "%.2f" % np.rad2deg(tc)
        print("    theta_c=%s deg: seq-ESN %.3f  ISAL-img %.3f" %
              (tcs_str, a, b))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(snrs, acc_snr, "o-")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("Robustness to noise (profile + ESN)")
    axes[0].grid(alpha=0.3)
    xlbl = ["inf" if np.isinf(t) else "%.2f" % np.rad2deg(t) for t in tcs_sweep]
    x = np.arange(len(tcs_sweep))
    axes[1].plot(x, acc_tc_esn, "o-", label="seq-ESN (incoherent sum)")
    axes[1].plot(x, acc_tc_img, "s-", label="ISAL image (coherent)")
    axes[1].set_xticks(x, xlbl)
    axes[1].set_xlabel("scatterer coherence angle (deg)")
    axes[1].set_ylabel("test accuracy")
    axes[1].set_title("Coherence-angle robustness")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "m4_fig_robustness.png"), dpi=130)
    plt.close(fig)

    # ---- Exp 5: small-target case (TFLN window) ---------------------------
    print("[exp5] small-target D=0.15 m ...")
    D2 = 0.15
    Ptr2 = profiles_dataset(Xtr, amp_tr, ph_tr, D2, sampler, seed_off=8000)
    Pte2 = profiles_dataset(Xte, amp_te, ph_te, D2, sampler, seed_off=9000)
    esn5 = ESN(1, n_res=512)
    acc_small, _ = ce_readout(esn5.features(Ptr2[:, :, None]), ytr,
                              esn5.features(Pte2[:, :, None]), yte, n_classes)
    print("    D=0.15 m profile+ESN: %.3f" % acc_small)

    # ---- save -------------------------------------------------------------
    np.savez(os.path.join(OUT, "m4_isal_results.npz"),
             names=names, cm=cm,
             acc_profile_ridge=acc["profile_ridge"],
             acc_profile_ce=acc["profile_ce"],
             acc_profile_esn_ce=acc["profile_esn_ce"],
             acc_isal_img_ce=acc["isal_img_ce"],
             acc_profile_cnn=acc["profile_cnn"],
             acc_4look_cnn=acc["4look_cnn"],
             acc_16look_cnn=acc["16look_cnn"],
             offsets=offsets, acc_vs_off=acc_vs_off,
             snrs=snrs, acc_snr=acc_snr,
             tcs_deg=[np.inf if np.isinf(t) else np.rad2deg(t)
                      for t in tcs_sweep],
             acc_tc_esn=acc_tc_esn, acc_tc_img=acc_tc_img,
             acc_small=acc_small)
    for f in ["m4_fig_profiles.png", "m4_fig_isal.png",
              "m4_fig_classification.png", "m4_fig_aspect.png",
              "m4_fig_robustness.png", "m4_isal_results.npz"]:
        shutil.copy(os.path.join(OUT, f), os.path.join(TFLN_RESULTS, f))
    print("[done] total %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
