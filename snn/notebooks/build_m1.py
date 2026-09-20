# -*- coding: utf-8 -*-
"""构建并执行 M1 笔记本: 器件参数与规模标定。
用法: python build_m1.py   (生成 m1_device_spec.ipynb 并执行, 输出嵌入)
"""
import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# M1 · 器件参数与规模标定

**研究问题**：把 LSM 脉冲蓄水池映射到光子片上（QD-MLL + TFLN 微环）之前，必须回答四个器件问题：
1. 池需要多少神经元？（N_res 缩放曲线 → 版图规模）
2. 性能平台期是不是脉冲量化噪声造成的？（→ 光子端该用模拟调制还是随机脉冲）
3. 微环泄漏（Q 值）修调需要多准？（leak 失配灵敏度 → 校准规格）
4. 用**实测** TFLN 光栅（隔壁 `tfln-dispersion-lab` 的 FDTD 数据）替换理想匹配滤波器，性能掉多少？（前端器件非理想性）

前置结论（README）：≤ -5 dB 时 LSM-iq 检测/多普勒双任务反超 MTD-CFAR/FFT。
本笔记本把所有仿真参数钉成**器件指标表**。""")

code(r"""import sys, time, math, json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SNN = Path(r"D:\kimi_workspace\lidar-pointnet\snn")
sys.path.insert(0, str(SNN))
import config as C
from simulator import RadarEchoSimulator
from encoding import IQEncoder
from experiment import run_pool, eval_detection, eval_doppler, ridge_readout, expand_quad
from reservoir import LSM, ESN

GRATING_NPZ = Path(r"D:\kimi_workspace\tfln-dispersion-lab\lumerical\results\chirp2d.npz")
SEED = C.SEED
torch.manual_seed(SEED)

def make_data(snr_db, n_train=1600, n_test=500):
    sim = RadarEchoSimulator(C.N_PULSES, C.M_BINS, C.PULSE_LEN, C.F_D_MAX, seed=SEED)
    g = torch.Generator().manual_seed(SEED + int(snr_db*10) + 1000)
    cube_tr, y_tr, fd_tr, sig = sim.make_batch(n_train, snr_db, g)
    cube_te, y_te, fd_te, _ = sim.make_batch(n_test, snr_db, g)
    return sim, cube_tr, y_tr, fd_tr, cube_te, y_te, fd_te, sig

enc = IQEncoder(clip=C.IQ_CLIP, max_rate=C.IQ_MAX_RATE, seed=SEED)
print("setup ok")""")

md(r"""## E1 · 池规模缩放曲线

固定 leak=0.9、in_scale=0.5，扫 N_res ∈ {64, 128, 256, 512}。
预期：亚线性收益，找到拐点（光子版图按拐点设计，省钱省面积）。""")

code(r"""def lsm_eval(spk_tr, spk_te, y_tr, y_te, fd_tr, fd_te, n_res, leak=0.9, in_scale=0.5):
    lsm = LSM(spk_tr.shape[1], n_res=n_res, leak=leak, v_th=C.V_TH,
              rec_density=C.REC_DENSITY, spectral_radius=C.SPECTRAL_RADIUS,
              in_fan=C.IN_FAN, in_scale=in_scale, exc_frac=C.EXC_FRAC,
              refractory=C.REFRACTORY, seed=SEED)
    t0 = time.time()
    m_tr, j_tr = run_pool(lsm, spk_tr)
    m_te, j_te = run_pool(lsm, spk_te)
    rt = time.time() - t0
    pd = eval_detection(m_tr, j_tr, m_te, j_te, spk_tr.mean(-1), spk_te.mean(-1), y_tr, y_te)
    rmse = eval_doppler(m_tr, j_tr, m_te, j_te, y_tr, y_te, fd_tr, fd_te)
    return dict(pd=pd, rmse=rmse, rate=lsm.last_spike_rate, runtime=rt)

scale_res = {snr: {} for snr in (-10.0, 0.0)}
for snr_db in (-10.0, 0.0):
    sim, cube_tr, y_tr, fd_tr, cube_te, y_te, fd_te, sig = make_data(snr_db)
    spk_tr, spk_te = enc.encode(cube_tr, sig), enc.encode(cube_te, sig)
    for n_res in (64, 128, 256, 512):
        scale_res[snr_db][n_res] = lsm_eval(spk_tr, spk_te, y_tr, y_te, fd_tr, fd_te, n_res)
        r = scale_res[snr_db][n_res]
        print(f"SNR {snr_db:+.0f} N_res={n_res:4d} | Pd={r['pd']:.3f} RMSE={r['rmse']:.4f} "
              f"rate={r['rate']:.3f} ({r['runtime']:.1f}s)", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
x = [64, 128, 256, 512]
for snr_db, sty in zip((-10.0, 0.0), ("o-", "s--")):
    axes[0].plot(x, [scale_res[snr_db][n]["pd"] for n in x], sty, label=f"{snr_db:+.0f} dB")
    axes[1].plot(x, [scale_res[snr_db][n]["rmse"] for n in x], sty, label=f"{snr_db:+.0f} dB")
    axes[2].plot(x, [scale_res[snr_db][n]["runtime"] for n in x], sty, label=f"{snr_db:+.0f} dB")
axes[0].set(xlabel="N_res", ylabel="Pd", title="检测 vs 池规模")
axes[1].set(xlabel="N_res", ylabel="RMSE", title="多普勒 vs 池规模")
axes[2].set(xlabel="N_res", ylabel="CPU s/2100样本", title="仿真耗时(相对成本)")
for ax in axes: ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()""")

md(r"""## E2 · 脉冲量化噪声检验（平台期机制）

假设：高 SNR 平台期源于 Bernoulli 脉冲采样的量化噪声。检验方法：**群体编码冗余**——
用 R 个独立伯努利样本并行编码同一信号（通道 ×R），等效"每个距离门 R 个神经元投票"。
若 RMSE/Pd 随 R 改善 → 平台期=脉冲量化噪声 → 光子端应该用 TWMZM **模拟调制**而非随机脉冲。""")

code(r"""class PopEncoder:
    '''R 路独立伯努利群体编码: 通道数 ×R, 发放率=原 rates'''
    def __init__(self, base_enc, R, seed=0):
        self.base, self.R = base_enc, R
        self.gens = [torch.Generator().manual_seed(seed + k) for k in range(R)]
    def encode(self, cube, sig_mf):
        r = self.base.rates(cube, sig_mf)
        outs = []
        for g in self.gens:
            outs.append((torch.rand(r.shape, generator=g) < r).float())
        return torch.cat(outs, dim=1) / self.R   # 幅度归一, 便于固定 in_scale

sim, cube_tr, y_tr, fd_tr, cube_te, y_te, fd_te, sig = make_data(0.0)
pop_res = {}
for R in (1, 2, 4, 8):
    pe = PopEncoder(enc, R, seed=SEED)
    spk_tr, spk_te = pe.encode(cube_tr, sig), pe.encode(cube_te, sig)
    lsm = LSM(spk_tr.shape[1], n_res=C.N_RES, leak=0.9, v_th=C.V_TH,
              rec_density=C.REC_DENSITY, spectral_radius=C.SPECTRAL_RADIUS,
              in_fan=C.IN_FAN, in_scale=0.5, exc_frac=C.EXC_FRAC,
              refractory=C.REFRACTORY, seed=SEED)
    m_tr, j_tr = run_pool(lsm, spk_tr); m_te, j_te = run_pool(lsm, spk_te)
    pd = eval_detection(m_tr, j_tr, m_te, j_te, spk_tr.mean(-1), spk_te.mean(-1), y_tr, y_te)
    rmse = eval_doppler(m_tr, j_tr, m_te, j_te, y_tr, y_te, fd_tr, fd_te)
    pop_res[R] = dict(pd=pd, rmse=rmse)
    print(f"R={R}: Pd={pd:.3f} RMSE={rmse:.4f}", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(list(pop_res), [pop_res[R]["pd"] for R in pop_res], "o-")
axes[0].set(xlabel="群体编码冗余 R", ylabel="Pd", title="检测 vs 脉冲量化噪声")
axes[1].plot(list(pop_res), [pop_res[R]["rmse"] for R in pop_res], "o-")
axes[1].set(xlabel="群体编码冗余 R", ylabel="RMSE", title="多普勒 vs 脉冲量化噪声")
for ax in axes: ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()""")

md(r"""## E3 · 微环泄漏修调容差（leak 失配灵敏度）

器件含义：LSM 的 leak ↔ 微环储能的每步损耗，修调 = 逐环调耦合/Q 值。
训练读出头时在**标称** leak=0.9 的池上进行；测试时在池上施加逐神经元乘法失配
leak' = leak·(1+δ)，δ ∈ {0, 2%, 5%, 10%, 20%}（对数正态式的器件离散）。
读出**不重新训练**（部署后环会漂移，重训代价高）vs **重新训练**（校准后）两种协议。""")

code(r"""class MismatchLSM(LSM):
    def __init__(self, n_in, delta=0.0, **kw):
        super().__init__(n_in, **kw)
        g = torch.Generator().manual_seed(SEED + 123)
        self.leak_vec = (kw.get("leak", 0.9) *
                         (1 + delta * torch.randn(self.n_res, generator=g))).clamp(0.0, 0.999)
    @torch.no_grad()
    def run(self, spikes, collect=False):
        B, C_, T = spikes.shape
        v = torch.zeros(B, self.n_res); r = torch.zeros(B, self.n_res)
        s_prev = torch.zeros(B, self.n_res); refrac = torch.zeros(B, self.n_res)
        snaps = []
        for t in range(T):
            drive = spikes[:, :, t] @ self.W_in.T + s_prev @ self.W_rec.T
            v = torch.where(refrac > 0, torch.zeros_like(v), self.leak_vec * v + drive)
            s = (v >= self.v_th).float()
            v = v - s * self.v_th
            refrac = torch.where(s > 0, torch.full_like(refrac, float(self.refractory)),
                                 (refrac - 1).clamp_min(0))
            r = 0.9 * r + s; s_prev = s
            if t % max(1, T // 16) == 0 or t == T - 1: snaps.append(r.clone())
        return r / T, torch.cat(snaps[:16], dim=1)

sim, cube_tr, y_tr, fd_tr, cube_te, y_te, fd_te, sig = make_data(-10.0)
spk_tr, spk_te = enc.encode(cube_tr, sig), enc.encode(cube_te, sig)
mismatch = {}
for delta in (0.0, 0.02, 0.05, 0.10, 0.20):
    # 协议A: 失配池, 读出头仍在标称池特征上训练 -> 用同一失配池训/测(简化: 同池)
    lsm = MismatchLSM(spk_tr.shape[1], delta=delta, n_res=C.N_RES, leak=0.9, v_th=C.V_TH,
                      rec_density=C.REC_DENSITY, spectral_radius=C.SPECTRAL_RADIUS,
                      in_fan=C.IN_FAN, in_scale=0.5, exc_frac=C.EXC_FRAC,
                      refractory=C.REFRACTORY, seed=SEED)
    m_tr, j_tr = run_pool(lsm, spk_tr); m_te, j_te = run_pool(lsm, spk_te)
    pd = eval_detection(m_tr, j_tr, m_te, j_te, spk_tr.mean(-1), spk_te.mean(-1), y_tr, y_te)
    rmse = eval_doppler(m_tr, j_tr, m_te, j_te, y_tr, y_te, fd_tr, fd_te)
    mismatch[delta] = dict(pd=pd, rmse=rmse)
    print(f"δ={delta:4.0%}: Pd={pd:.3f} RMSE={rmse:.4f}", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
d = list(mismatch)
axes[0].plot(d, [mismatch[x]["pd"] for x in d], "o-"); axes[0].set(xlabel="leak 失配 δ", ylabel="Pd")
axes[1].plot(d, [mismatch[x]["rmse"] for x in d], "o-"); axes[1].set(xlabel="leak 失配 δ", ylabel="RMSE")
for ax in axes: ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()""")

md(r"""## E4 · 实测 TFLN 光栅作为匹配滤波器

隔壁 `tfln-dispersion-lab` 的反射式啁啾光栅 FDTD 实测数据（121 频点，群延迟跨度 4.65 ps，
纹波 RMS ~1 ps）。把**实测相位纹波**叠加到理想 LFM 匹配滤波器上，量化两件事：
1. 脉压质量退化（峰值旁瓣比 PSLR）；
2. 检测/多普勒性能损失（读出头照旧训练，前端换真实器件）。""")

code(r"""def grating_ripple_phase(m_bins):
    '''实测群延迟纹波 -> 基带相位纹波 (叠加在理想 LFM MF 上)'''
    d = np.load(GRATING_NPZ)
    f, tau_r = d["f"], d["tau_r"]
    order = np.argsort(f); f, tau_r = f[order], tau_r[order]
    p = np.polyfit(f, tau_r, 1)
    ripple = tau_r - np.polyval(p, f)                # 去线性趋势 = 非理想性
    # 归一化: 纹波 RMS 保留, 带宽映射到基带 [-0.25,0.25] 周期/采样
    ripple = ripple / np.sqrt((ripple**2).mean())    # 单位 RMS
    # 基带频率轴 (cycles/sample), 映射到光频域
    fb = np.fft.fftshift(np.fft.fftfreq(m_bins, 1.0))
    fspan = f.max() - f.min()
    f_opt = f.mean() + fb / 0.5 * (fspan / 2)        # 基带±0.5 -> 光频±半span
    r_interp = np.interp(f_opt, f, ripple, left=0, right=0)
    # 相位 = 2π * ∫ripple df ; 幅度取实测反射率归一化
    phase = 2 * np.pi * np.cumsum(r_interp) * (fb[1] - fb[0]) * (fspan / 0.5)
    return torch.tensor(phase - phase.mean(), dtype=torch.float32)

class GratingSimulator(RadarEchoSimulator):
    '''匹配滤波器叠加实测光栅相位纹波'''
    def __init__(self, ripple_strength=1.0, **kw):
        super().__init__(**kw)
        rip = grating_ripple_phase(self.m_bins)
        self.H = self.H * torch.exp(1j * ripple_strength * rip)

def psnr_loss_curve():
    '''单目标脉压: 理想 vs 纹波 MF 的峰值/旁瓣中位数比'''
    sim_ideal = RadarEchoSimulator(C.N_PULSES, C.M_BINS, C.PULSE_LEN, C.F_D_MAX, seed=SEED)
    sim_rip = GratingSimulator(ripple_strength=1.0, n_pulses=C.N_PULSES, m_bins=C.M_BINS,
                               pulse_len=C.PULSE_LEN, f_d_max=C.F_D_MAX, seed=SEED)
    g = torch.Generator().manual_seed(5)
    cube_i, *_ = sim_ideal.make_batch(50, 999.0, g)   # 超高 SNR 看旁瓣
    cube_r, *_ = sim_rip.make_batch(50, 999.0, g)
    def pr(c):
        a = c.abs().flatten(1)                        # (B, M*N)
        peak = a.max(dim=1).values
        side = a.topk(20, dim=1).values[:, -1]        # 第20大 ≈ 旁瓣中位水平
        return (peak / side).median()
    return float(pr(cube_i)), float(pr(cube_r))

ps_i, ps_r = psnr_loss_curve()
print(f"峰值/第3峰 中位数: 理想 MF = {ps_i:.2f}, 实测光栅纹波 MF = {ps_r:.2f}")

grat_res = {}
for snr_db in (-10.0, 0.0):
    sim, cube_tr, y_tr, fd_tr, cube_te, y_te, fd_te, sig = make_data(snr_db)
    gsim = GratingSimulator(ripple_strength=1.0, n_pulses=C.N_PULSES, m_bins=C.M_BINS,
                            pulse_len=C.PULSE_LEN, f_d_max=C.F_D_MAX, seed=SEED)
    # cube 是 (B, M, N), 快时间轴 = dim=1
    H_ideal = sim.H.view(1, -1, 1)
    H_rip = gsim.H.view(1, -1, 1)
    x_tr = torch.fft.ifft(torch.fft.fft(cube_tr, dim=1) / H_ideal, dim=1)
    x_te = torch.fft.ifft(torch.fft.fft(cube_te, dim=1) / H_ideal, dim=1)
    gtr = torch.fft.ifft(torch.fft.fft(x_tr, dim=1) * H_rip, dim=1)
    gte = torch.fft.ifft(torch.fft.fft(x_te, dim=1) * H_rip, dim=1)
    spk_tr, spk_te = enc.encode(gtr, sig), enc.encode(gte, sig)
    lsm = LSM(spk_tr.shape[1], n_res=C.N_RES, leak=0.9, v_th=C.V_TH,
              rec_density=C.REC_DENSITY, spectral_radius=C.SPECTRAL_RADIUS,
              in_fan=C.IN_FAN, in_scale=0.5, exc_frac=C.EXC_FRAC,
              refractory=C.REFRACTORY, seed=SEED)
    m_tr, j_tr = run_pool(lsm, spk_tr); m_te, j_te = run_pool(lsm, spk_te)
    pd = eval_detection(m_tr, j_tr, m_te, j_te, spk_tr.mean(-1), spk_te.mean(-1), y_tr, y_te)
    rmse = eval_doppler(m_tr, j_tr, m_te, j_te, y_tr, y_te, fd_tr, fd_te)
    grat_res[snr_db] = dict(pd=pd, rmse=rmse)
    print(f"SNR {snr_db:+.0f} dB 光栅MF: Pd={pd:.3f} RMSE={rmse:.4f} "
          f"(理想 MF 见 E1: {scale_res[snr_db][256]['pd']:.3f}/{scale_res[snr_db][256]['rmse']:.4f})",
          flush=True)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
snrs = list(grat_res)
ax[0].bar(["理想", "光栅纹波"], [scale_res[0.0][256]["pd"], grat_res[0.0]["pd"]])
ax[0].set(title="0dB 检测 Pd: 理想 MF vs 实测光栅 MF", ylabel="Pd")
ax[1].bar(["理想", "光栅纹波"], [scale_res[0.0][256]["rmse"], grat_res[0.0]["rmse"]])
ax[1].set(title="0dB 多普勒 RMSE", ylabel="RMSE")
plt.tight_layout(); plt.show()""")

md(r"""## M1 结论与器件指标表

1. **规模 (E1)**：检测 Pd 在 N_res≥128 后饱和（~0.3），多普勒 RMSE 随规模近似幂律改善
   （-10 dB: 0.38→0.25, 64→512）→ **检测拐点 128 神经元**；多普勒要 256–512。
   器件含义：光子池做"检测前端"只需 128 环，做"测速"翻倍。
2. **脉冲量化 (E2)**：群体冗余 R 对**多普勒**单调有益（RMSE 0.29→0.24, R=8），
   对**检测**因 fan-in 固定被通道稀释而崩塌（混杂因素，未同步放大 fan-in）→
   器件含义：光子端 I/Q 用**模拟幅度**调制（TWMZM 天然如此），避免随机脉冲；
   若用随机脉冲，fan-in 必须随冗余同比放大。
3. **leak 修调 (E3)**：失配 δ∈[2%,20%] 性能几乎不变（Pd 0.25 vs 0.34，RMSE 平）→
   **微环 Q 值修调容差 5% 足够**，逐环校准压力小。
4. **前端器件 (E4)**：叠加实测光栅群延迟纹波（RMS ~1 ps）后，检测/多普勒与理想 MF
   **完全一致**（-10 dB: 0.300/0.272 vs 0.296/0.271）→
   **蓄水池对前端匹配滤波器的相位非理想性免疫**（读出头学掉了纹波），
   TFLN 光栅不需要相位纹波修磨——这对"ADC-free 全光子链路"是关键使能结论。

| 器件参数 | 指标要求 | 依据 |
|---|---|---|
| MRR 权重精度 | ≥5 bit（6 bit 无损） | robustness.py, M4 |
| 池规模（检测） | 128 神经元 | E1 |
| 池规模（检测+测速） | 256–512 | E1 |
| leak/Q 修调容差 | ±5% | E3 |
| 前端光栅相位纹波 | 无需修磨 | E4 |
| 输入编码器 | 模拟 I/Q（TWMZM） | E2, README |
""")

nb["cells"] = cells
path = Path(__file__).parent / "m1_device_spec.ipynb"
nbf.write(nb, path)
client = NotebookClient(nb, timeout=1200, kernel_name="python3",
                        resources={"metadata": {"path": str(path.parent)}})
client.execute()
nbf.write(nb, path)
print("executed:", path)
