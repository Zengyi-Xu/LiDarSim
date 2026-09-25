# -*- coding: utf-8 -*-
"""
M4-extension: automotive-style coarse classification from a single forward
range profile.

The user's point: real driving scenes don't have multi-angle beams; the task
is coarse classes (vehicle / pedestrian / rider / background) whose 1D
silhouettes are very different. This script validates that intuition using
ModelNet40-like classes.

Tasks:
  A) 2-class: car vs non-car
  B) 2-class: vehicle (car+airplane) vs non-vehicle
  C) 4-class coarse: car, airplane, chair, bottle  (very different shapes)
  D) 10-class fine (for reference)
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
    get_data, make_scatterers, complex_profiles, ESN, ce_readout,
    standardize, N_RNG, SEED
)

OUT = r"D:\kimi_workspace\lidar-pointnet\snn\outputs_isal"
TFLN_RESULTS = r"D:\kimi_workspace\tfln-dispersion-lab\results\snn"


def profiles_random(X, amp, ph0, D, seed_off=0):
    out = np.zeros((len(X), N_RNG), dtype=np.float32)
    for i in range(len(X)):
        rng = np.random.default_rng(SEED + 60000 + seed_off + i)
        th = np.array([rng.uniform(0, 2 * np.pi)])
        out[i] = np.abs(complex_profiles(X[i], amp[i], ph0[i], th, D,
                                         np.inf, rng)[0])
    return out


def run_task(name, Xtr, ytr, Xte, yte, n_classes, seed_off):
    print("  %-22s n=%d acc=" % (name, n_classes), end="")
    Pte = profiles_random(Xte, amp_te, ph_te, D, seed_off=seed_off)
    esn = ESN(1, n_res=512)
    acc, _ = ce_readout(
        esn.features(Ptr[:, :, None]), ytr,
        esn.features(Pte[:, :, None]), yte, n_classes)
    print("%.3f" % acc)
    return acc


def main():
    t0 = time.time()
    global Ptr, D, amp_tr, ph_tr, amp_te, ph_te
    print("[load] data ...")
    Xtr10, ytr10, Xte10, yte10, classes10 = get_data(n_classes=10)
    names10 = ["airplane", "bathtub", "bed", "bench", "bookshelf",
               "bottle", "bowl", "car", "chair", "cone"]
    D = 4.0
    amp_tr, ph_tr = make_scatterers(Xtr10, SEED)
    amp_te, ph_te = make_scatterers(Xte10, SEED + 1)

    # single random profiles for the full 10-class set
    Ptr = profiles_random(Xtr10, amp_tr, ph_tr, D, seed_off=0)

    results = {}

    # ---- A) car vs non-car ------------------------------------------------
    ci = names10.index("car")
    mask_tr = (ytr10 == ci)
    mask_te = (yte10 == ci)
    yA_tr = mask_tr.astype(int)
    yA_te = mask_te.astype(int)
    results["car_vs_noncar"] = run_task("car vs non-car",
                                        Xtr10, yA_tr, Xte10, yA_te, 2, 10)

    # ---- B) vehicle (car+airplane) vs non-vehicle -------------------------
    ai, ci = names10.index("airplane"), names10.index("car")
    yB_tr = ((ytr10 == ai) | (ytr10 == ci)).astype(int)
    yB_te = ((yte10 == ai) | (yte10 == ci)).astype(int)
    results["vehicle_vs_nonvehicle"] = run_task(
        "vehicle vs non-vehicle", Xtr10, yB_tr, Xte10, yB_te, 2, 20)

    # ---- C) 4-class coarse: car, airplane, chair, bottle ------------------
    class_map = {names10.index("car"): 0,
                 names10.index("airplane"): 1,
                 names10.index("chair"): 2,
                 names10.index("bottle"): 3}
    idx_tr = np.isin(ytr10, list(class_map.keys()))
    idx_te = np.isin(yte10, list(class_map.keys()))
    XtrC, ytrC = Xtr10[idx_tr], np.vectorize(class_map.get)(ytr10[idx_tr])
    XteC, yteC = Xte10[idx_te], np.vectorize(class_map.get)(yte10[idx_te])
    amp_trC, ph_trC = amp_tr[idx_tr], ph_tr[idx_tr]
    amp_teC, ph_teC = amp_te[idx_te], ph_te[idx_te]
    PtrC = profiles_random(XtrC, amp_trC, ph_trC, D, seed_off=30)
    PteC = profiles_random(XteC, amp_teC, ph_teC, D, seed_off=31)
    esnC = ESN(1, n_res=512)
    results["coarse_4class"], _ = ce_readout(
        esnC.features(PtrC[:, :, None]), ytrC,
        esnC.features(PteC[:, :, None]), yteC, 4)
    print("  %-22s n=%d acc=%.3f" % ("coarse 4-class", 4,
                                     results["coarse_4class"]))

    # ---- D) 10-class reference --------------------------------------------
    Pte10 = profiles_random(Xte10, amp_te, ph_te, D, seed_off=40)
    esn10 = ESN(1, n_res=512)
    results["fine_10class"], _ = ce_readout(
        esn10.features(Ptr[:, :, None]), ytr10,
        esn10.features(Pte10[:, :, None]), yte10, 10)
    print("  %-22s n=%d acc=%.3f" % ("fine 10-class", 10,
                                     results["fine_10class"]))

    # ---- plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    keys = ["car_vs_noncar", "vehicle_vs_nonvehicle",
            "coarse_4class", "fine_10class"]
    vals = [results[k] for k in keys]
    cols = ["C2", "C2", "C1", "C0"]
    ax.bar(range(len(keys)), vals, color=cols)
    ax.set_xticks(range(len(keys)),
                  ["car vs\nnon-car", "vehicle vs\nnon-vehicle",
                   "coarse\n4-class", "fine\n10-class"])
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Single random HRRP: coarse vs fine-grained tasks")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, "%.2f" % v, ha="center", fontsize=10)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "m4_fig_automotive_focus.png"), dpi=130)
    plt.close(fig)

    np.savez(os.path.join(OUT, "m4_automotive_focus_results.npz"), **results)
    for f in ["m4_fig_automotive_focus.png", "m4_automotive_focus_results.npz"]:
        shutil.copy(os.path.join(OUT, f), os.path.join(TFLN_RESULTS, f))
    print("[done] %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
