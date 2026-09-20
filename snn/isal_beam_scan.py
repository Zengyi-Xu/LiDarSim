# -*- coding: utf-8 -*-
"""
M4-extension: realistic multi-beam / sector-limited ordered scanning.

Scenario:
  - 4 fixed beams (or lenses) oriented at known angles, e.g. -45, -15, +15, +45 deg.
  - Each beam performs a small ordered sweep around its centre (sector-limited scan).
  - The emitter angle is known, so profiles can be fed in a structured order
    and/or with angle appended as an auxiliary channel.

Compare against:
  - single random-aspect profile (M4 baseline)
  - 4 random looks averaged (M4 multi-look baseline)
  - 4 fixed beams: incoherent average, concat, ESN sequence
  - 4 beams x 3-step sweep: ESN sequence

This tests the user's hypothesis: realistic structured illumination should
recover a large part of the information lost in the random single-shot case.
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

from isal_range_profile import (
    get_data, make_scatterers, complex_profiles, ESN, ce_readout,
    standardize, N_RNG, SEED, OUT as OUT_ROOT
)

OUT = OUT_ROOT
TFLN_RESULTS = r"D:\kimi_workspace\tfln-dispersion-lab\results\snn"


def beam_profiles(X, amp, ph0, D, beam_centers, sweep_deg, n_steps,
                  add_angle=False, snr_db=None, seed_off=0):
    """Generate ordered (n_samples, n_beams*n_steps, n_in) profile sequence.

    beam_centers: list of angles in radians (known emission angles)
    sweep_deg: half-width of sweep around each centre, in degrees
    n_steps: number of ordered steps per beam (must be odd to include centre)
    add_angle: if True, append sin(theta), cos(theta) as extra channels
    """
    sweep = np.deg2rad(sweep_deg)
    steps = np.linspace(-sweep, sweep, n_steps)
    n_beams = len(beam_centers)
    n_in = N_RNG + (2 if add_angle else 0)
    seq = np.zeros((len(X), n_beams * n_steps, n_in), dtype=np.float32)
    for i in range(len(X)):
        rng = np.random.default_rng(SEED + 10000 + seed_off + i)
        t = 0
        for c, bc in enumerate(beam_centers):
            for s in steps:
                th = np.array([bc + s])
                prof = np.abs(complex_profiles(X[i], amp[i], ph0[i], th, D,
                                               np.inf, rng, snr_db)[0])
                if add_angle:
                    seq[i, t, :N_RNG] = prof
                    seq[i, t, N_RNG] = np.sin(th[0])
                    seq[i, t, N_RNG + 1] = np.cos(th[0])
                else:
                    seq[i, t] = prof
                t += 1
    return seq


def random_looks(X, amp, ph0, D, n_looks, snr_db=None, seed_off=0):
    """n_looks random-aspect magnitude profiles per sample, averaged."""
    out = np.zeros((len(X), N_RNG), dtype=np.float32)
    for i in range(len(X)):
        rng = np.random.default_rng(SEED + 20000 + seed_off + i)
        s = np.zeros(N_RNG, dtype=np.float32)
        for _ in range(n_looks):
            th = np.array([rng.uniform(0, 2 * np.pi)])
            s += np.abs(complex_profiles(X[i], amp[i], ph0[i], th, D,
                                         np.inf, rng, snr_db)[0])
        out[i] = s / n_looks
    return out


def cnn1d_acc(Xtr, ytr, Xte, yte, n_classes, epochs=300, lr=3e-3, seed=SEED):
    """Small 1D CNN from M4 benchmark."""
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


def strong_2d_cnn_acc(Xtr, ytr, Xte, yte, n_classes, epochs=400, lr=1e-3,
                      seed=SEED):
    """Deep 2D CNN on (n_steps, n_rng) scan map as empirical upper bound."""
    torch.manual_seed(seed)
    # per-channel standardization over all spatial positions
    mu = Xtr.mean(axis=(0, 1), keepdims=True)
    sd = Xtr.std(axis=(0, 1), keepdims=True) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr_t = torch.tensor(Xtr[:, None, :, :], dtype=torch.float32)
    Xte_t = torch.tensor(Xte[:, None, :, :], dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    yte_t = torch.tensor(yte, dtype=torch.long)
    T, R = Xtr.shape[1], Xtr.shape[2]
    net = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
        nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
        nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
        nn.ReLU(), nn.AdaptiveAvgPool2d((4, 8)),
        nn.Flatten(),
        nn.Linear(128 * 4 * 8, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, n_classes))
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


def main():
    t0 = time.time()
    print("[load] data ...")
    Xtr, ytr, Xte, yte, classes = get_data()
    names = ["airplane", "bathtub", "bed", "bench", "bookshelf",
             "bottle", "bowl", "car", "chair", "cone"]
    n_classes = len(classes)
    D = 4.0
    amp_tr, ph_tr = make_scatterers(Xtr, SEED)
    amp_te, ph_te = make_scatterers(Xte, SEED + 1)
    print("  train %d test %d classes" % (len(Xtr), len(Xte)))

    acc = {}

    # ---- M4 baselines (same setting) ------------------------------------
    print("[baseline] single random profile ...")
    from isal_range_profile import profiles_dataset
    P1tr = profiles_dataset(Xtr, amp_tr, ph_tr, D,
                            lambda rng: rng.uniform(0, 2 * np.pi), seed_off=0)
    P1te = profiles_dataset(Xte, amp_te, ph_te, D,
                            lambda rng: rng.uniform(0, 2 * np.pi), seed_off=1)
    esn1 = ESN(1, n_res=512)
    acc["single_random_ESN"], _ = ce_readout(
        esn1.features(P1tr[:, :, None]), ytr,
        esn1.features(P1te[:, :, None]), yte, n_classes)
    acc["single_random_CNN"] = cnn1d_acc(P1tr, ytr, P1te, yte, n_classes)

    print("[baseline] 4 random looks average ...")
    R4tr = random_looks(Xtr, amp_tr, ph_tr, D, 4, seed_off=2)
    R4te = random_looks(Xte, amp_te, ph_te, D, 4, seed_off=3)
    acc["4random_avg_CNN"] = cnn1d_acc(R4tr, ytr, R4te, yte, n_classes)

    # ---- 4 fixed beams, snapshot (1 profile per beam) --------------------
    beam_centers = np.deg2rad([-45, -15, 15, 45])
    print("[exp] 4 fixed beams snapshot ...")
    B4tr = beam_profiles(Xtr, amp_tr, ph_tr, D, beam_centers, 0, 1,
                         add_angle=False, seed_off=4)
    B4te = beam_profiles(Xte, amp_te, ph_te, D, beam_centers, 0, 1,
                         add_angle=False, seed_off=5)
    # incoherent average of 4 beams
    B4avg_tr = B4tr.mean(axis=1)
    B4avg_te = B4te.mean(axis=1)
    acc["4beam_avg_CNN"] = cnn1d_acc(B4avg_tr, ytr, B4avg_te, yte, n_classes)
    # concat 4 beams
    B4cat_tr = B4tr.reshape(len(B4tr), -1)
    B4cat_te = B4te.reshape(len(B4te), -1)
    acc["4beam_concat_CE"], _ = ce_readout(B4cat_tr, ytr, B4cat_te, yte,
                                           n_classes)
    # ESN over ordered 4-beam sequence
    esn4 = ESN(N_RNG, n_res=512)
    acc["4beam_seq_ESN"], _ = ce_readout(
        esn4.features(B4tr), ytr, esn4.features(B4te), yte, n_classes)
    # ESN with known angle appended
    B4a_tr = beam_profiles(Xtr, amp_tr, ph_tr, D, beam_centers, 0, 1,
                           add_angle=True, seed_off=4)
    B4a_te = beam_profiles(Xte, amp_te, ph_te, D, beam_centers, 0, 1,
                           add_angle=True, seed_off=5)
    esn4a = ESN(N_RNG + 2, n_res=512)
    acc["4beam_seq_ESN+angle"], _ = ce_readout(
        esn4a.features(B4a_tr), ytr, esn4a.features(B4a_te), yte, n_classes)

    # ---- 4 beams x 3-step sector scan ------------------------------------
    print("[exp] 4 beams x 3-step ordered sweep ...")
    B12tr = beam_profiles(Xtr, amp_tr, ph_tr, D, beam_centers, 3, 3,
                          add_angle=False, seed_off=6)
    B12te = beam_profiles(Xte, amp_te, ph_te, D, beam_centers, 3, 3,
                          add_angle=False, seed_off=7)
    esn12 = ESN(N_RNG, n_res=512)
    acc["4beam_3step_ESN"], _ = ce_readout(
        esn12.features(B12tr), ytr, esn12.features(B12te), yte, n_classes)

    # also test same 12 profiles randomly shuffled (control: order matters?)
    B12rand_tr = B12tr.copy()
    B12rand_te = B12te.copy()
    for i in range(len(B12rand_tr)):
        np.random.default_rng(SEED + 30000 + i).shuffle(B12rand_tr[i])
    for i in range(len(B12rand_te)):
        np.random.default_rng(SEED + 31000 + i).shuffle(B12rand_te[i])
    esn12r = ESN(N_RNG, n_res=512)
    acc["4beam_3step_shuffled_ESN"], _ = ce_readout(
        esn12r.features(B12rand_tr), ytr, esn12r.features(B12rand_te),
        yte, n_classes)

    # ---- empirical ceiling: deep 2D CNN on the scan map ------------------
    print("[exp] deep 2D CNN ceiling on 4beam x 3step scan map ...")
    acc["4beam_3step_2DCNN_ceiling"] = strong_2d_cnn_acc(
        B12tr, ytr, B12te, yte, n_classes)

    # ---- 2 beams and 8 beams scaling -------------------------------------
    print("[exp] beam count scaling ...")
    for n_b, label in [(2, "2beam_seq_ESN"), (8, "8beam_seq_ESN")]:
        if n_b == 2:
            centers = np.deg2rad([-30, 30])
        else:
            centers = np.deg2rad(np.linspace(-60, 60, 8))
        Btr = beam_profiles(Xtr, amp_tr, ph_tr, D, centers, 0, 1,
                            add_angle=False, seed_off=8 + n_b)
        Bte = beam_profiles(Xte, amp_te, ph_te, D, centers, 0, 1,
                            add_angle=False, seed_off=9 + n_b)
        esn_b = ESN(N_RNG, n_res=512)
        acc[label], _ = ce_readout(esn_b.features(Btr), ytr,
                                   esn_b.features(Bte), yte, n_classes)

    # ---- print & plot ----------------------------------------------------
    print("\nResults:")
    for k, v in acc.items():
        print("  %-28s %.3f" % (k, v))

    fig, ax = plt.subplots(figsize=(9, 5))
    keys = list(acc.keys())
    vals = list(acc.values())
    bars = ax.bar(range(len(keys)), vals)
    bars[0].set_color("C0")
    bars[1].set_color("C0")
    bars[2].set_color("C1")
    for i in [3, 4, 5, 6, 7, 8, 9, 10]:
        bars[i].set_color("C2")
    ax.set_xticks(range(len(keys)), keys, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Multi-beam sector-limited ordered scanning (D=4 m)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, "%.2f" % v, ha="center", fontsize=7)
    ax.axhline(0.1, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "m4_fig_beam_scan.png"), dpi=130)
    plt.close(fig)

    np.savez(os.path.join(OUT, "m4_beam_scan_results.npz"),
             B12tr=B12tr, B12te=B12te, ytr=ytr, yte=yte, **acc)
    for f in ["m4_fig_beam_scan.png", "m4_beam_scan_results.npz"]:
        shutil.copy(os.path.join(OUT, f), os.path.join(TFLN_RESULTS, f))
    print("[done] %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
