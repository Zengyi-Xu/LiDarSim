"""主实验: 低 SNR 检测 + 多普勒估计
LSM (三种编码, 验证集网格搜索 leak/输入增益) / ESN 基线 vs MTD-CFAR / FFT, 扫描 SNR。

读出头设计 (关键):
    检测   = 对 [池均值, 输入发放率] 做二次展开 [X, X^2] —— 未知距离门的
             目标检测是能量统计量, 必须在随机混合之前的通道上取平方;
    多普勒 = 对 [状态轨迹快照, 池均值] 线性回归 —— 相位进动信息在
             状态的时间轨迹里, 线性读出头沿时间加权即类 DFT 投影。
所有检测器统一协议: 门限在训练集负样本上按 P_FA=1e-3 标定。
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import config as C
from simulator import RadarEchoSimulator
from encoding import EventEncoder, IQEncoder, DeltaEncoder
from reservoir import LSM, ESN
import cfar

CHUNK = 1000
VAL_FRAC = 0.2
LSM_GRID = [(leak, ins) for leak in (0.9, 0.97) for ins in (0.2, 0.5)]


def ridge_readout(Xtr, ytr, Xte, alpha=C.RIDGE_ALPHA):
    """标准化 + 带偏置脊回归, 返回 (测试预测, 训练预测)。

    训练集近零方差的特征直接丢弃, 避免测试集偶发放大导致数值爆炸。
    """
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True)
    keep = (sd >= 1e-2).squeeze(0)
    Xtr = (Xtr - mu) / sd.clamp_min(1e-2)
    Xte = (Xte - mu) / sd.clamp_min(1e-2)
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    Xtr = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1)], dim=1)
    Xte = torch.cat([Xte, torch.ones(Xte.shape[0], 1)], dim=1)
    A = Xtr.T @ Xtr + alpha * torch.eye(Xtr.shape[1])
    W = torch.linalg.solve(A, Xtr.T @ ytr)
    return Xte @ W, Xtr @ W


def expand_quad(X):
    return torch.cat([X, X * X], dim=1)


def run_pool(pool, u, collect=False):
    """分块前向, 避免大批次内存峰值。返回 (mean, traj) 或 (mean, traj, trace)。"""
    outs = []
    for i in range(0, u.shape[0], CHUNK):
        if collect:
            outs.append(pool.run(u[i:i + CHUNK], collect=True))
        else:
            outs.append(pool.run(u[i:i + CHUNK]))
    n = len(outs[0])
    result = []
    for j in range(n):
        result.append(torch.cat([o[j] for o in outs]))
    return result[0] if n == 1 else tuple(result)


def eval_detection(mean_tr, traj_tr, mean_te, traj_te, in_tr, in_te, ytr, yte):
    """检测 = 能量统计量。轨迹末快照 r(T) 是 leaky 积分器的相干积累输出
    (神经元群体随机滤波器 + 平方读出 ~= 周期图检测器); 池均值/输入发放率
    提供非相干能量。二次展开 + 脊回归, 阈值取训练负样本 (1-P_FA) 分位。"""
    tail_tr, tail_te = traj_tr[:, -C.N_RES:], traj_te[:, -C.N_RES:]
    Ftr = expand_quad(torch.cat([tail_tr, mean_tr, in_tr], dim=1))
    Fte = expand_quad(torch.cat([tail_te, mean_te, in_te], dim=1))
    _, score_tr = ridge_readout(Ftr, ytr.float(), Ftr[:1])
    thr = torch.quantile(score_tr[~ytr.bool()], 1 - C.P_FA)
    score_te, _ = ridge_readout(Ftr, ytr.float(), Fte)
    return float((score_te[yte.bool()] > thr).float().mean())


def eval_doppler(mean_tr, traj_tr, mean_te, traj_te, ytr, yte, fdtr, fdte):
    """轨迹 + 池均值线性回归 (仅目标样本)。特征维度高, 用较强正则。"""
    tr_idx, te_idx = ytr.bool(), yte.bool()
    Ftr = torch.cat([traj_tr, mean_tr], dim=1)[tr_idx]
    Fte = torch.cat([traj_te, mean_te], dim=1)[te_idx]
    pred, _ = ridge_readout(Ftr, fdtr[tr_idx].unsqueeze(1), Fte, alpha=10.0)
    return float(torch.sqrt(torch.mean((pred.squeeze(1) - fdte[te_idx]) ** 2)))


def select_lsm(n_in, spk_tr, spk_val, ytr, yval, fdtr, fdval):
    """小网格搜索蓄水池 leak/输入增益, 验证集打分, 返回最优池与其训练态。"""
    best = None
    fallback = None
    in_tr, in_val = spk_tr.mean(-1), spk_val.mean(-1)
    for leak, ins in LSM_GRID:
        lsm = LSM(n_in, n_res=C.N_RES, leak=leak, v_th=C.V_TH,
                  rec_density=C.REC_DENSITY, spectral_radius=C.SPECTRAL_RADIUS,
                  in_fan=C.IN_FAN, in_scale=ins, exc_frac=C.EXC_FRAC,
                  refractory=C.REFRACTORY, seed=C.SEED)
        m_tr, j_tr = run_pool(lsm, spk_tr)
        m_val, j_val = run_pool(lsm, spk_val)
        rate = lsm.last_spike_rate
        cand = {"leak": leak, "in_scale": ins, "lsm": lsm,
                "m_tr": m_tr, "j_tr": j_tr, "res_spk": rate}
        if fallback is None:
            fallback = cand
        if rate > 0.5 or rate < 1e-4:            # 池饱和/死寂: 动态无效, 淘汰
            continue
        pd_v = eval_detection(m_tr, j_tr, m_val, j_val, in_tr, in_val, ytr, yval)
        rmse_v = eval_doppler(m_tr, j_tr, m_val, j_val, ytr, yval, fdtr, fdval)
        score = pd_v + max(0.0, 1.0 - rmse_v / 0.05) * 0.2      # 检测为主, 多普勒为辅
        if best is None or score > best["score"]:
            best = dict(cand, score=score)
    return best if best is not None else dict(fallback, score=float("-inf"))


def demo_figure(sim, lsm, enc, sigma_mf, snr_db, out_path):
    """单目标示例: 距离-多普勒图 / 输入脉冲光栅 / 池内脉冲光栅。"""
    g = torch.Generator().manual_seed(C.SEED + 999)
    cube, _, fd, _ = sim.make_batch(2, snr_db, g)
    cube, fd = cube[1:], fd[1:]
    rd = torch.fft.fftshift(torch.fft.fft(cube, dim=-1), dim=-1).abs()[0]
    spikes = enc.encode(cube, sigma_mf)
    _, _, trace = run_pool(lsm, spikes, collect=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    im = axes[0].imshow(20 * torch.log10(rd / rd.max() + 1e-6),
                        aspect="auto", origin="lower", cmap="viridis",
                        extent=(-0.5, 0.5, 0, C.M_BINS))
    axes[0].axvline(float(fd[0]), color="r", ls="--", lw=1)
    axes[0].set_title(f"Range-Doppler (MTD), true $f_D$={float(fd[0]):.3f}")
    axes[0].set_xlabel("Doppler (cycles/pulse)")
    axes[0].set_ylabel("Range bin")
    fig.colorbar(im, ax=axes[0])
    axes[1].spy(spikes[0], markersize=0.5, aspect="auto")
    axes[1].set_title(f"Input spikes ({enc.name})")
    axes[1].set_xlabel("Pulse index")
    axes[2].spy(trace[0], markersize=0.5, aspect="auto")
    axes[2].set_title("LSM reservoir spikes")
    axes[2].set_xlabel("Pulse index")
    fig.suptitle(f"SNR = {snr_db:.0f} dB (MF output peak, per pulse)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=C.N_TRAIN)
    ap.add_argument("--n-test", type=int, default=C.N_TEST)
    ap.add_argument("--snr", type=float, nargs="+", default=C.SNR_LIST_DB)
    ap.add_argument("--out", type=Path, default=C.OUTPUT_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(C.SEED)

    sim = RadarEchoSimulator(C.N_PULSES, C.M_BINS, C.PULSE_LEN, C.F_D_MAX, seed=C.SEED)
    encoders = {
        "event": EventEncoder(C.EVENT_THRESH),
        "iq": IQEncoder(clip=C.IQ_CLIP, max_rate=C.IQ_MAX_RATE, seed=C.SEED),
        "delta": DeltaEncoder(C.DELTA_THRESH),
    }
    keys = list(encoders.keys()) + ["esn-iq"]
    res = {k: {"pd": [], "rmse": [], "in_spk": [], "res_spk": [], "hp": []} for k in keys}
    base = {"cfar_pd": [], "fft_rmse": [], "crlb": []}
    demo_done = False

    for snr_db in args.snr:
        t0 = time.time()
        g = torch.Generator().manual_seed(C.SEED + int(snr_db * 10) + 1000)
        cube_tr, y_tr, fd_tr, sig_mf = sim.make_batch(args.n_train, snr_db, g)
        cube_te, y_te, fd_te, _ = sim.make_batch(args.n_test, snr_db, g)

        perm = torch.randperm(args.n_train, generator=g)
        n_val = int(args.n_train * VAL_FRAC)
        idx_tr, idx_val = perm[n_val:], perm[:n_val]

        for name, enc in encoders.items():
            spk_tr = enc.encode(cube_tr, sig_mf)
            spk_te = enc.encode(cube_te, sig_mf)
            best = select_lsm(spk_tr.shape[1],
                              spk_tr[idx_tr], spk_tr[idx_val],
                              y_tr[idx_tr], y_tr[idx_val],
                              fd_tr[idx_tr], fd_tr[idx_val])
            m_te, j_te = run_pool(best["lsm"], spk_te)
            pd = eval_detection(best["m_tr"], best["j_tr"], m_te, j_te,
                                spk_tr[idx_tr].mean(-1), spk_te.mean(-1),
                                y_tr[idx_tr], y_te)
            rmse = eval_doppler(best["m_tr"], best["j_tr"], m_te, j_te,
                                y_tr[idx_tr], y_te, fd_tr[idx_tr], fd_te)
            res[name]["pd"].append(pd)
            res[name]["rmse"].append(rmse)
            res[name]["in_spk"].append(float(spk_te.mean()))
            res[name]["res_spk"].append(best["res_spk"])
            res[name]["hp"].append([best["leak"], best["in_scale"]])
            if name == "iq" and not demo_done:
                demo_figure(sim, best["lsm"], enc, sig_mf, snr_db, args.out / "demo_signals.png")
                demo_done = True

        esn = ESN(n_in=encoders["iq"].encode(cube_te[:1], sig_mf).shape[1],
                  n_res=C.ESN_RES, leak=C.ESN_LEAK, in_scale=C.ESN_IN_SCALE,
                  rec_density=C.ESN_REC_DENSITY,
                  spectral_radius=C.ESN_SPECTRAL_RADIUS, seed=C.SEED)
        iq_rates_tr = encoders["iq"].rates(cube_tr, sig_mf)
        iq_rates_te = encoders["iq"].rates(cube_te, sig_mf)
        em_tr, ej_tr = run_pool(esn, iq_rates_tr)
        em_te, ej_te = run_pool(esn, iq_rates_te)
        pd = eval_detection(em_tr, ej_tr, em_te, ej_te,
                            iq_rates_tr.mean(-1), iq_rates_te.mean(-1), y_tr, y_te)
        rmse = eval_doppler(em_tr, ej_tr, em_te, ej_te, y_tr, y_te, fd_tr, fd_te)
        res["esn-iq"]["pd"].append(pd)
        res["esn-iq"]["rmse"].append(rmse)
        res["esn-iq"]["in_spk"].append(float(iq_rates_te.mean()))
        res["esn-iq"]["res_spk"].append(float("nan"))
        res["esn-iq"]["hp"].append([C.ESN_LEAK, C.ESN_IN_SCALE])

        ratio_tr, valid = cfar.mtd_cfar_ratio(cube_tr[~y_tr])
        thr = cfar.calibrate_threshold(ratio_tr, valid, C.P_FA)
        ratio_te, _ = cfar.mtd_cfar_ratio(cube_te)
        dec = cfar.mtd_cfar_detect(ratio_te, valid, thr)
        base["cfar_pd"].append(float(dec[y_te.bool()].float().mean()))

        fd_hat = cfar.fft_doppler_est(cube_te[y_te.bool()])
        base["fft_rmse"].append(float(torch.sqrt(torch.mean((fd_hat - fd_te[y_te.bool()]) ** 2))))
        base["crlb"].append(cfar.crlb_doppler(10.0 ** (snr_db / 10), C.N_PULSES))

        print(f"SNR {snr_db:+5.1f} dB | " + " | ".join(
            f"{k}: Pd={res[k]['pd'][-1]:.3f}" for k in keys)
            + f" | CFAR Pd={base['cfar_pd'][-1]:.3f} | {time.time()-t0:.0f}s", flush=True)

    json.dump({"snr_db": args.snr, "p_fa": C.P_FA, "res": res, "base": base},
              open(args.out / "results.json", "w"), indent=1)
    plot(args.snr, res, base, args.out)
    print(f"\n结果已保存到 {args.out}")


def plot(snr_db, res, base, out_dir):
    colors = {"event": "tab:blue", "iq": "tab:red", "delta": "tab:green", "esn-iq": "tab:orange"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for k, v in res.items():
        ls = "--" if k == "esn-iq" else "-"
        label = f"LSM-{k}" if k != "esn-iq" else "ESN-iq"
        axes[0].plot(snr_db, v["pd"], ls, color=colors[k], marker="o", label=label)
        axes[1].plot(snr_db, v["rmse"], ls, color=colors[k], marker="o", label=label)
        axes[2].plot(snr_db, v["in_spk"], ls, color=colors[k], marker="o", label=k)
    axes[0].plot(snr_db, base["cfar_pd"], "k--", marker="s", label="MTD-CFAR (classic)")
    axes[1].plot(snr_db, base["fft_rmse"], "k--", marker="s", label="FFT (classic)")
    axes[1].plot(snr_db, base["crlb"], "k:", label="CRLB")

    axes[0].set(title="Detection", ylabel=f"Pd (Pfa={C.P_FA:g})", xlabel="MF peak SNR (dB/pulse)")
    axes[1].set(title="Doppler estimation", ylabel="RMSE (cycles/pulse)",
                xlabel="MF peak SNR (dB/pulse)", yscale="log")
    axes[2].set(title="Input spike cost", ylabel="spikes / (bin·pulse)",
                xlabel="MF peak SNR (dB/pulse)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "summary.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
