# -*- coding: utf-8 -*-
"""3D 视角采样扫描仿真: 赤道环带 (road_crossing) vs 上半球冠 (uav_cap).

扩展 isal_range_profile 的面内方位角扫描为完整 3D 旋转:
LOS(az, el) = (cos el·cos az, cos el·sin az, sin el),  r = P · LOS.
(z 维在原面内版本里不参与投影, 本脚本用完整 3D 旋转采样)

场景预设 (与真实相对几何一致):
  road_crossing: az~U(0,2π), el~U(-5°,5°)    路侧/车载, 路口任意航向
  uav_cap:       az~U(0,2π), el~U(25°,65°)   UAV 俯视交通监管

实验 (全部带图, 图存 outputs_isal/3dview/):
  E0 视线流形示意图 (3D 球面散点) + 类内随方位角的剖面变化
  E1 每个场景: 单次 HRRP+CNN1D vs 4 波束x3 步扫描+ESN (± 角度先验)
  E2 跨分布泛化: road 训 -> uav 测 / 反向 (两种臂)
  E3 俯仰维标价: 固定 12 剖面预算, 赤道环 vs 双环 vs 随机半球
"""
import os, sys, time, json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
OUT = HERE / "outputs_isal" / "3dview"
OUT.mkdir(parents=True, exist_ok=True)

from isal_range_profile import (get_data, make_scatterers, ou_phase_walk,
                                ESN, ce_readout, cnn1d_acc, N_RNG, SEED)

D = 4.0
SCEN = {"road_crossing": (-5.0, 5.0), "uav_cap": (25.0, 65.0)}  # el 范围 (度)


# ---------------- 3D 剖面生成 ----------------
def profiles_3d(P, amp, ph0, views, D, theta_c=np.inf, rng=None, snr_db=None):
    """views: (M,2) (az, el) rad. 返回复数剖面 (M, N_RNG)."""
    if rng is None:
        rng = np.random.default_rng(0)
    cell = D / N_RNG
    lam = cell / 8.0
    xyz = P * D / 2.0                                          # (Pp,3)
    az, el = views[:, 0], views[:, 1]
    los = np.stack([np.cos(el) * np.cos(az),
                    np.cos(el) * np.sin(az), np.sin(el)], 1)   # (M,3)
    r = (los @ xyz.T)                                          # (M, Pp)
    if len(views) > 1:
        dcos = np.clip((los[:-1] * los[1:]).sum(1), -1, 1)
        dth = float(np.arccos(dcos).mean()) or 1.0
    else:
        dth = 1.0
    ou = ou_phase_walk(len(views), dth, theta_c, (len(xyz),), rng)
    phase = 4 * np.pi * r / lam + ph0[None, :] + np.pi * ou
    contrib = amp[None, :] * np.exp(1j * phase)
    idx = np.clip(np.floor((r + D / 2) / cell).astype(int), 0, N_RNG - 1)
    prof = np.zeros((len(views), N_RNG), dtype=np.complex64)
    midx = np.repeat(np.arange(len(views)), len(xyz))
    np.add.at(prof, (midx, idx.ravel()), contrib.ravel())
    if snr_db is not None:
        p_sig = np.mean(np.abs(prof) ** 2)
        p_n = p_sig / (10 ** (snr_db / 10))
        prof = prof + np.sqrt(p_n / 2) * (
            rng.standard_normal(prof.shape) + 1j * rng.standard_normal(prof.shape))
    return prof


def sample_views(scenario, rng):
    """单次随机视角 (az, el) rad"""
    el_lo, el_hi = np.deg2rad(SCEN[scenario])
    return np.array([rng.uniform(0, 2 * np.pi),
                     rng.uniform(el_lo, el_hi)])


def scan_views(scenario, n_beams=4, sweep_deg=3.0, n_steps=3):
    """4 波束 (方位 -45..45) x 3 步小扇区扫描的视角序列 (T,2); el 取场景中值"""
    el_mid = np.deg2rad(np.mean(SCEN[scenario]))
    az_c = np.deg2rad(np.linspace(-45, 45, n_beams))
    offs = np.deg2rad(np.linspace(-sweep_deg / 2, sweep_deg / 2, n_steps))
    views = []
    for a in az_c:
        for o in offs:
            views.append([a + o, el_mid])
    return np.array(views)


def make_dataset(X, amp, ph0, scenario, kind, seed_off=0, add_angle=False):
    """kind: 'single' -> (N, N_RNG); 'scan' -> (N, T, N_RNG) (+可选角度通道)"""
    views_scan = scan_views(scenario)
    T = len(views_scan)
    n_in = N_RNG + (3 if add_angle else 0)
    out_s = np.zeros((len(X), N_RNG), dtype=np.float32)
    out_q = np.zeros((len(X), T, n_in), dtype=np.float32)
    for i in range(len(X)):
        rng = np.random.default_rng(SEED + 1000 + seed_off + i)
        v = sample_views(scenario, rng)
        p = profiles_3d(X[i], amp[i], ph0[i], v[None, :], D, rng=rng)
        out_s[i] = np.abs(p[0])
        rng2 = np.random.default_rng(SEED + 2000 + seed_off + i)
        jit = rng2.normal(0, np.deg2rad(0.3), views_scan.shape)
        q = profiles_3d(X[i], amp[i], ph0[i], views_scan + jit, D, rng=rng2)
        mag = np.abs(q)
        if add_angle:
            los = np.stack([np.cos(views_scan[:, 1]) * np.cos(views_scan[:, 0]),
                            np.cos(views_scan[:, 1]) * np.sin(views_scan[:, 0]),
                            np.sin(views_scan[:, 1])], 1)
            mag = np.concatenate([mag, los], 1)
        out_q[i] = mag
    return out_s, out_q


# ---------------- 实验 ----------------
def e0_figures(X, amp, ph0, y):
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 60); v = np.linspace(-np.pi / 2, np.pi / 2, 30)
    ax.plot_wireframe(np.outer(np.cos(u), np.cos(v)), np.outer(np.sin(u), np.cos(v)),
                      np.outer(np.ones(len(u)), np.sin(v)), color="0.88", lw=0.3)
    for name, c in [("road_crossing", "C0"), ("uav_cap", "C3")]:
        rng = np.random.default_rng(0)
        views = np.stack([sample_views(name, rng) for _ in range(300)])
        s = np.stack([np.cos(views[:, 1]) * np.cos(views[:, 0]),
                      np.cos(views[:, 1]) * np.sin(views[:, 0]),
                      np.sin(views[:, 1])], 1)
        ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=2, c=c, label=name, alpha=0.6)
    ax.set(xlabel="x", ylabel="y", zlabel="z", title="E0 视线流形: 赤道环带 vs 上半球冠")
    ax.set_box_aspect([1, 1, 0.9]); ax.legend()
    plt.tight_layout(); plt.savefig(OUT / "e0_view_manifolds.png", dpi=130); plt.close()

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 6, figsize=(13, 4.8), sharex=True)
    for row, (k, name) in enumerate([(7, "car"), (5, "bottle")]):
        idx = np.where(y == k)[0][0]
        azs = np.deg2rad(np.arange(0, 360, 60))
        views = np.stack([azs, np.zeros(6)], 1)
        p = np.abs(profiles_3d(X[idx], amp[idx], ph0[idx], views, D, rng=rng))
        for j in range(6):
            axes[row, j].plot(p[j], lw=1.2)
            axes[row, j].set_title(f"{name} az={np.rad2deg(azs[j]):.0f}°", fontsize=9)
            axes[row, j].grid(alpha=0.3)
    fig.suptitle("E0 类内剖面随方位角的变化 (el=0, 赤道环)", fontsize=11)
    plt.tight_layout(); plt.savefig(OUT / "e0_aspect_variation.png", dpi=130); plt.close()
    print("[E0] figures saved: view manifolds + aspect variation", flush=True)


def e1(Xtr, ytr, Xte, yte, amp_tr, ph_tr, amp_te, ph_te, n_classes):
    res = {}
    for scenario in SCEN:
        print(f"[E1] {scenario} ...", flush=True)
        s_tr, q_tr = make_dataset(Xtr, amp_tr, ph_tr, scenario,
                                  "scan", seed_off=0, add_angle=True)
        s_te, q_te = make_dataset(Xte, amp_te, ph_te, scenario,
                                  "scan", seed_off=1, add_angle=True)
        row = {}
        row["单次HRRP+CNN1D"] = cnn1d_acc(s_tr, ytr, s_te, yte,
                                          n_classes, epochs=200)
        esn = ESN(N_RNG + 3, n_res=512)
        row["4波束x3步+ESN+角度"] = ce_readout(
            esn.features(q_tr), ytr, esn.features(q_te), yte,
            n_classes, epochs=200)[0]
        res[scenario] = row
        for k, v in row.items():
            print(f"   {k}: {v:.3f}", flush=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(2); w = 0.35
    for j, arm in enumerate(["单次HRRP+CNN1D", "4波束x3步+ESN+角度"]):
        ax.bar(x + j * w, [res[s][arm] for s in SCEN], w, label=arm)
    ax.set_xticks(x + w / 2, list(SCEN)); ax.axhline(0.1, color="k", ls=":",
                                                     label="随机(10类)")
    ax.set_ylabel("测试准确率"); ax.set_title("E1 场景内分类: 单次 vs 扫描链")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(OUT / "e1_scenario_acc.png", dpi=130); plt.close()
    return res


def e2(Xtr, ytr, Xte, yte, amp_tr, ph_tr, amp_te, ph_te, n_classes):
    print("[E2] 跨分布泛化 ...", flush=True)
    s_road_tr, _ = make_dataset(Xtr, amp_tr, ph_tr, "road_crossing",
                                "single", seed_off=10)
    s_road_te, _ = make_dataset(Xte, amp_te, ph_te, "road_crossing",
                                "single", seed_off=11)
    s_uav_tr, _ = make_dataset(Xtr, amp_tr, ph_tr, "uav_cap",
                               "single", seed_off=10)
    s_uav_te, _ = make_dataset(Xte, amp_te, ph_te, "uav_cap",
                               "single", seed_off=11)
    M = np.zeros((2, 2))
    M[0, 0] = cnn1d_acc(s_road_tr, ytr, s_road_te, yte, n_classes, epochs=200)
    M[0, 1] = cnn1d_acc(s_road_tr, ytr, s_uav_te, yte, n_classes, epochs=200)
    M[1, 1] = cnn1d_acc(s_uav_tr, ytr, s_uav_te, yte, n_classes, epochs=200)
    M[1, 0] = cnn1d_acc(s_uav_tr, ytr, s_road_te, yte, n_classes, epochs=200)
    print("   [road->road %.3f, road->uav %.3f, uav->uav %.3f, uav->road %.3f]"
          % (M[0, 0], M[0, 1], M[1, 1], M[1, 0]), flush=True)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks([0, 1], ["road 测", "uav 测"]); ax.set_yticks([0, 1], ["road 训", "uav 训"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center",
                    color="w", fontsize=12)
    ax.set_title("E2 跨分布泛化 (单次HRRP+CNN1D)")
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout(); plt.savefig(OUT / "e2_cross_dist.png", dpi=130); plt.close()
    return M.tolist()


def e3(Xtr, ytr, Xte, yte, amp_tr, ph_tr, amp_te, ph_te, n_classes):
    print("[E3] 俯仰维标价 (12 剖面预算) ...", flush=True)
    configs = {
        "赤道环 12az (el=0)": np.stack([np.deg2rad(np.arange(0, 360, 30)),
                                        np.zeros(12)], 1),
        "双环 2x6az (el=10/55)": np.concatenate([
            np.stack([np.deg2rad(np.arange(0, 360, 60)),
                      np.full(6, np.deg2rad(10.))], 1),
            np.stack([np.deg2rad(np.arange(0, 360, 60)),
                      np.full(6, np.deg2rad(55.))], 1)]),
        "随机半球 12 (el∈[10,80])": None,
    }
    res = {}
    ntr, nte = len(Xtr), len(Xte)
    for name, views in configs.items():
        tr = np.zeros((ntr, 12, N_RNG), dtype=np.float32)
        te = np.zeros((nte, 12, N_RNG), dtype=np.float32)
        for split, (X_, A_, P_, buf) in enumerate(
                [(Xtr, amp_tr, ph_tr, tr), (Xte, amp_te, ph_te, te)]):
            for j in range(len(X_)):
                rng = np.random.default_rng(SEED + 5000 + split * 50000 + j)
                vw = views
                if vw is None:
                    vw = np.stack([rng.uniform(0, 2 * np.pi, 12),
                                   rng.uniform(np.deg2rad(10), np.deg2rad(80), 12)], 1)
                buf[j] = np.abs(profiles_3d(X_[j], A_[j], P_[j], vw, D, rng=rng))
        esn = ESN(N_RNG, n_res=512)
        res[name] = ce_readout(esn.features(tr), ytr,
                               esn.features(te), yte, n_classes, epochs=200)[0]
        print(f"   {name}: {res[name]:.3f}", flush=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(res), list(res.values()), color=["C0", "C2", "C3"])
    ax.axvline(0.1, color="k", ls=":", label="随机(10类)")
    for i, (k, v) in enumerate(res.items()):
        ax.text(v + 0.005, i, "%.3f" % v, va="center", fontsize=10)
    ax.set_xlabel("测试准确率"); ax.set_title("E3 俯仰维标价 (固定 12 剖面预算)")
    ax.set_xlim(0, 1); ax.legend(); ax.grid(alpha=0.3, axis="x")
    plt.tight_layout(); plt.savefig(OUT / "e3_elevation_price.png", dpi=130); plt.close()
    return res


def main():
    t0 = time.time()
    print("[load] data ...", flush=True)
    Xtr, ytr, Xte, yte, classes = get_data()
    n_classes = len(classes)
    amp_tr, ph_tr = make_scatterers(Xtr, SEED)
    amp_te, ph_te = make_scatterers(Xte, SEED + 1)
    print("  %d train / %d test, %d classes" % (len(Xtr), len(Xte), n_classes),
          flush=True)
    e0_figures(Xtr, amp_tr, ph_tr, ytr)
    r1 = e1(Xtr, ytr, Xte, yte, amp_tr, ph_tr, amp_te, ph_te, n_classes)
    r2 = e2(Xtr, ytr, Xte, yte, amp_tr, ph_tr, amp_te, ph_te, n_classes)
    r3 = e3(Xtr, ytr, Xte, yte, amp_tr, ph_tr, amp_te, ph_te, n_classes)
    results = {"E1": r1, "E2": r2, "E3": r3}
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    np.savez(OUT / "results.npz", **{"E1": np.array([[r1[s][a] for a in
             ("单次HRRP+CNN1D", "4波束x3步+ESN+角度")] for s in SCEN]),
             "E2": np.array(r2), "E3": np.array(list(r3.values()))})
    print("done in %.1f s; results -> %s" % (time.time() - t0, OUT), flush=True)


if __name__ == "__main__":
    main()
