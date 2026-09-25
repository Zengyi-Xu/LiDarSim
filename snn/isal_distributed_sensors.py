# -*- coding: utf-8 -*-
"""
M4-extension: distributed multi-sensor (multi-static) HRRP classification.

Compare:
  - 4 beams from one location at different steering angles (collocated)
  - 4 sensors physically separated around the target (distributed)
    * equatorial square (0/90/180/270 deg on a circle)
    * tetrahedral (4 corners of a tetrahedron around target)

Each sensor is monostatic, at known position, looking toward target centre.
Range profiles are aligned to the target-centred window [-D/2, D/2].
"""
import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isal_range_profile import (
    get_data, make_scatterers, ESN, ce_readout, N_RNG, SEED
)
from isal_beam_scan import cnn1d_acc, strong_2d_cnn_acc

OUT = r"D:\kimi_workspace\lidar-pointnet\snn\outputs_isal"
TFLN_RESULTS = r"D:\kimi_workspace\tfln-dispersion-lab\results\snn"


def distributed_profiles(X, amp, ph0, D, sensor_positions,
                         add_angle=False, snr_db=None, seed_off=0):
    """Range profiles from spatially distributed monostatic sensors.

    X: (N, P, 3) point clouds normalised to max(|coord|)=1.
    sensor_positions: (S, 3) sensor coordinates (m), target at origin.
    Returns (N, S, n_in) float32.
    """
    n_sensors = len(sensor_positions)
    n_in = N_RNG + (2 if add_angle else 0)
    seq = np.zeros((len(X), n_sensors, n_in), dtype=np.float32)
    cell = D / N_RNG
    lam = cell / 8.0
    for i in range(len(X)):
        rng = np.random.default_rng(SEED + 40000 + seed_off + i)
        pts = X[i] * D / 2.0                # metres, target spans ~D
        for s, spos in enumerate(sensor_positions):
            view = -spos / (np.linalg.norm(spos) + 1e-12)  # looks to origin
            r_rel = pts @ view                            # centred at 0
            idx = np.floor((r_rel + D / 2) / cell).astype(int)
            idx = np.clip(idx, 0, N_RNG - 1)
            phase = 4 * np.pi * r_rel / lam + ph0[i]
            contrib = amp[i] * np.exp(1j * phase)
            prof = np.zeros(N_RNG, dtype=np.complex64)
            np.add.at(prof, idx, contrib)
            if snr_db is not None:
                p_sig = np.mean(np.abs(prof) ** 2)
                p_n = p_sig / (10 ** (snr_db / 10))
                noise = np.sqrt(p_n / 2) * (
                    rng.standard_normal(N_RNG) +
                    1j * rng.standard_normal(N_RNG))
                prof = prof + noise
            if add_angle:
                seq[i, s, :N_RNG] = np.abs(prof)
                seq[i, s, N_RNG] = np.sin(np.arctan2(view[1], view[0]))
                seq[i, s, N_RNG + 1] = np.cos(np.arctan2(view[1], view[0]))
            else:
                seq[i, s] = np.abs(prof)
    return seq


def beam_from_angles(X, amp, ph0, D, angles_deg, seed_off=0):
    """Reuse isal_range_profile.complex_profiles for collocated beams."""
    from isal_range_profile import complex_profiles
    n_beams = len(angles_deg)
    seq = np.zeros((len(X), n_beams, N_RNG), dtype=np.float32)
    for i in range(len(X)):
        rng = np.random.default_rng(SEED + 50000 + seed_off + i)
        for b, deg in enumerate(angles_deg):
            th = np.array([np.deg2rad(deg)])
            seq[i, b] = np.abs(complex_profiles(X[i], amp[i], ph0[i], th, D,
                                                np.inf, rng)[0])
    return seq


def main():
    t0 = time.time()
    print("[load] data ...")
    Xtr, ytr, Xte, yte, classes = get_data()
    n_classes = len(classes)
    D = 6.0  # window large enough to hold diagonal extent of 4 m target
    amp_tr, ph_tr = make_scatterers(Xtr, SEED)
    amp_te, ph_te = make_scatterers(Xte, SEED + 1)
    print("  train %d test %d classes, window D=%.1f m" % (len(Xtr), len(Xte), D))

    acc = {}

    R = 10.0  # sensor standoff distance (m)

    # ---- collocated 4 beams (baseline from beam_scan) --------------------
    print("[exp] collocated 4 beams snapshot ...")
    Btr = beam_from_angles(Xtr, amp_tr, ph_tr, D, [-45, -15, 15, 45],
                           seed_off=0)
    Bte = beam_from_angles(Xte, amp_te, ph_te, D, [-45, -15, 15, 45],
                           seed_off=1)
    acc["collocated_4beam_avg_CNN"] = cnn1d_acc(
        Btr.mean(axis=1), ytr, Bte.mean(axis=1), yte, n_classes)
    acc["collocated_4beam_concat_CE"], _ = ce_readout(
        Btr.reshape(len(Btr), -1), ytr,
        Bte.reshape(len(Bte), -1), yte, n_classes)
    esn_b = ESN(N_RNG, n_res=512)
    acc["collocated_4beam_seq_ESN"], _ = ce_readout(
        esn_b.features(Btr), ytr, esn_b.features(Bte), yte, n_classes)

    # ---- distributed: equatorial square ---------------------------------
    print("[exp] distributed equatorial 4 sensors ...")
    eq_pos = np.array([[R, 0, 0],
                       [0, R, 0],
                       [-R, 0, 0],
                       [0, -R, 0]], dtype=np.float32)
    Eqtr = distributed_profiles(Xtr, amp_tr, ph_tr, D, eq_pos, seed_off=2)
    Eqte = distributed_profiles(Xte, amp_te, ph_te, D, eq_pos, seed_off=3)
    acc["equatorial_avg_CNN"] = cnn1d_acc(
        Eqtr.mean(axis=1), ytr, Eqte.mean(axis=1), yte, n_classes)
    acc["equatorial_concat_CE"], _ = ce_readout(
        Eqtr.reshape(len(Eqtr), -1), ytr,
        Eqte.reshape(len(Eqte), -1), yte, n_classes)
    esn_eq = ESN(N_RNG, n_res=512)
    acc["equatorial_seq_ESN"], _ = ce_readout(
        esn_eq.features(Eqtr), ytr, esn_eq.features(Eqte), yte, n_classes)

    # ---- distributed: tetrahedral ---------------------------------------
    print("[exp] distributed tetrahedral 4 sensors ...")
    # vertices of regular tetrahedron centred at origin
    a = R / np.sqrt(3)
    tet_pos = np.array([[a, a, a],
                        [a, -a, -a],
                        [-a, a, -a],
                        [-a, -a, a]], dtype=np.float32)
    Ttr = distributed_profiles(Xtr, amp_tr, ph_tr, D, tet_pos, seed_off=4)
    Tte = distributed_profiles(Xte, amp_te, ph_te, D, tet_pos, seed_off=5)
    acc["tetrahedral_avg_CNN"] = cnn1d_acc(
        Ttr.mean(axis=1), ytr, Tte.mean(axis=1), yte, n_classes)
    acc["tetrahedral_concat_CE"], _ = ce_readout(
        Ttr.reshape(len(Ttr), -1), ytr,
        Tte.reshape(len(Tte), -1), yte, n_classes)
    esn_tet = ESN(N_RNG, n_res=512)
    acc["tetrahedral_seq_ESN"], _ = ce_readout(
        esn_tet.features(Ttr), ytr, esn_tet.features(Tte), yte, n_classes)

    # ---- empirical ceiling on best distributed representation -----------
    print("[exp] 2D CNN ceiling on equatorial 4-sensor scan map ...")
    acc["equatorial_2DCNN_ceiling"] = strong_2d_cnn_acc(
        Eqtr, ytr, Eqte, yte, n_classes)
    print("[exp] 2D CNN ceiling on tetrahedral 4-sensor scan map ...")
    acc["tetrahedral_2DCNN_ceiling"] = strong_2d_cnn_acc(
        Ttr, ytr, Tte, yte, n_classes)

    # ---- print & plot ---------------------------------------------------
    print("\nResults:")
    for k, v in acc.items():
        print("  %-32s %.3f" % (k, v))

    keys = ["collocated_4beam_avg_CNN", "collocated_4beam_concat_CE",
            "collocated_4beam_seq_ESN",
            "equatorial_avg_CNN", "equatorial_concat_CE",
            "equatorial_seq_ESN",
            "tetrahedral_avg_CNN", "tetrahedral_concat_CE",
            "tetrahedral_seq_ESN", "tetrahedral_2DCNN_ceiling"]
    vals = [acc[k] for k in keys]
    colors = (["C0"] * 3 + ["C1"] * 3 + ["C2"] * 3 + ["C3"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(keys)), vals, color=colors)
    ax.set_xticks(range(len(keys)), keys, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Collocated vs distributed sensors (4 views, D=4 m)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, "%.2f" % v, ha="center", fontsize=7)
    ax.axhline(0.1, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "m4_fig_distributed.png"), dpi=130)
    plt.close(fig)

    np.savez(os.path.join(OUT, "m4_distributed_results.npz"), **acc)
    for f in ["m4_fig_distributed.png", "m4_distributed_results.npz"]:
        shutil.copy(os.path.join(OUT, f), os.path.join(TFLN_RESULTS, f))
    print("[done] %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
