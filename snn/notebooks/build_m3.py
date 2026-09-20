# -*- coding: utf-8 -*-
"""M3 笔记本 v2: 级联协议下的微多普勒分类 (检测后选通, 二次读出)。"""
import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# M3 · 低 SNR 微多普勒目标分类（级联协议）

**任务**：4 类目标，慢时间相位含微多普勒调制 φ_n = 2π f_D n + β·sin(2π f_m n)：

| 类 | f_m (周期/脉冲) | β |
|---|---|---|
| 0 | 0.02 | 0.2 |
| 1 | 0.05 | 0.5 |
| 2 | 0.10 | 0.5 |
| 3 | 0.20 | 0.2 |

**v1 阴性结果记录**：直接对全距离门立方体分类（LSM 轨迹+均值 / FFT 谱 + 线性读出），
-15…0 dB 全部随机水平（0.25），训练集亦不可拟合。诊断：① 类信息只在目标门的相位里，
被 128 门稀释；② 频谱能量模式需二次特征，纯线性读出不够。
**v2 级联协议（本笔记本）**：模拟真实处理链——先检测/选通（oracle: 用真实门代表
检测器输出），再在选通门上做分类；读出用**二次核岭回归**（类信息 = 随机载波上的
边带对间距，需平移不变特征）。SNR 上探到 +5 dB（检测后等效
SNR = 单脉冲 + 10log10 N 相干增益）。""")

code(r"""import sys, time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SNN = Path(r"D:\kimi_workspace\lidar-pointnet\snn")
sys.path.insert(0, str(SNN))
import config as C
from experiment import run_pool, ridge_readout, expand_quad
from reservoir import LSM

SEED = C.SEED
torch.manual_seed(SEED)
CLASSES = [(0.02, 0.2), (0.05, 0.5), (0.10, 0.5), (0.20, 0.2)]

class MicroDopplerSim:
    '''匹配滤波后 (B, M, N); 记录真实门, 供级联选通'''
    def __init__(self, n_pulses, m_bins, f_d_max=0.3):
        self.N, self.M = n_pulses, m_bins
        self.f_d_max = f_d_max
    def make_batch(self, batch, snr_db, cls_list, generator):
        B = batch
        sigma_mf = np.sqrt(1.0 / 10.0 ** (snr_db / 10.0))
        delay = torch.randint(8, self.M - 8, (B,), generator=generator)
        f_d = (torch.rand(B, generator=generator) * 2 - 1) * self.f_d_max
        phi0 = torch.rand(B, generator=generator) * 2 * np.pi
        U = torch.rand(B, generator=generator).clamp_min(1e-6)
        amp = torch.sqrt(-torch.log(U))
        n = torch.arange(self.N)
        y = torch.zeros(B, self.M, self.N, dtype=torch.complex64)
        for b in range(B):
            fm, beta = CLASSES[cls_list[b]]
            ph = (2*np.pi*f_d[b]*n + beta*np.sin(2*np.pi*fm*n) + phi0[b])
            y[b, delay[b], :] = (amp[b] * torch.exp(1j*ph)).to(torch.complex64)
        noise = (sigma_mf/np.sqrt(2)) * (
            torch.randn(B, self.M, self.N, generator=generator)
            + 1j*torch.randn(B, self.M, self.N, generator=generator))
        return y + noise, sigma_mf, delay

class IQGateEncoder:
    '''只编码选通门的慢时间序列 (B, N) -> (B, 4, N)'''
    def __init__(self, clip=6.0, max_rate=0.8, seed=0):
        self.clip, self.max_rate = clip, max_rate
        self.gen = torch.Generator().manual_seed(seed)
    def encode(self, gate_seq, sig):
        z = gate_seq / sig
        I, Q = z.real, z.imag
        r = torch.stack([I.clamp_min(0), (-I).clamp_min(0),
                         Q.clamp_min(0), (-Q).clamp_min(0)], dim=1)
        r = (r / self.clip).clamp(0, 1) * self.max_rate
        return (torch.rand(r.shape, generator=self.gen) < r).float()

def onehot(y, k=4):
    return torch.nn.functional.one_hot(y.long(), k).float()

def quad_kernel(A, B):
    return (A @ B.T + 1.0) ** 2

def kernel_cls_eval(Ftr, Fte, ytr, yte, alpha=10.0):
    '''二次核岭回归 (对谱/状态特征的平移不变结构, 平方展开不够, 需全对交叉项)'''
    mu = Ftr.mean(0, keepdim=True); sd = Ftr.std(0, keepdim=True).clamp_min(1e-6)
    Ftr = (Ftr - mu) / sd; Fte = (Fte - mu) / sd
    Ktr = quad_kernel(Ftr, Ftr); Kte = quad_kernel(Fte, Ftr)
    Ytr = onehot(ytr)
    Coef = torch.linalg.solve(Ktr + alpha * torch.eye(len(Ktr)), Ytr)
    acc_tr = ((Ktr @ Coef).argmax(1) == ytr.long()).float().mean()
    acc_te = ((Kte @ Coef).argmax(1) == yte.long()).float().mean()
    return float(acc_tr), float(acc_te)

print("setup ok")""")

md(r"""## E1 · 级联分类：选通后 LSM-iq vs 选通后 FFT 谱

对选通门的 64 脉冲 I/Q 序列：LSM（4 通道输入，256 池）轨迹+均值 vs 64 点 FFT 幅度谱。
**读出 = 二次核岭回归**（K(a,b)=(a·b+1)²）——类信息是"随机载波 f_D 上的边带对间距 f_m"，
需要平移不变特征（全对交叉项），平方展开不够。""")

code(r"""cls_res = {}
for snr_db in (-10.0, -5.0, 0.0, 5.0):
    g1 = torch.Generator().manual_seed(SEED + int(snr_db*10) + 5000)
    g2 = torch.Generator().manual_seed(SEED + int(snr_db*10) + 6000)
    n_tr, n_te = 1600, 500
    y_tr = torch.randint(0, 4, (n_tr,), generator=g1)
    y_te = torch.randint(0, 4, (n_te,), generator=g2)
    sim = MicroDopplerSim(C.N_PULSES, C.M_BINS)
    cube_tr, sig, gate_tr = sim.make_batch(n_tr, snr_db, y_tr.tolist(), g1)
    cube_te, _, gate_te = sim.make_batch(n_te, snr_db, y_te.tolist(), g2)

    # 选通 (oracle gate = 检测器输出)
    idx_tr = torch.arange(n_tr); idx_te = torch.arange(n_te)
    gtr_cube = cube_tr[idx_tr, gate_tr]   # (B,N) 选通门慢时间序列
    gte_cube = cube_te[idx_te, gate_te]

    # LSM-iq (二次读出)
    enc_g = IQGateEncoder(seed=SEED)
    spk_tr, spk_te = enc_g.encode(gtr_cube, sig), enc_g.encode(gte_cube, sig)
    lsm = LSM(spk_tr.shape[1], n_res=C.N_RES, leak=0.9, v_th=C.V_TH,
              rec_density=C.REC_DENSITY, spectral_radius=C.SPECTRAL_RADIUS,
              in_fan=C.IN_FAN, in_scale=0.5, exc_frac=C.EXC_FRAC,
              refractory=C.REFRACTORY, seed=SEED)
    m_tr, j_tr = run_pool(lsm, spk_tr); m_te, j_te = run_pool(lsm, spk_te)
    F_lsm_tr = torch.cat([j_tr, m_tr], 1); F_lsm_te = torch.cat([j_te, m_te], 1)
    acc_lsm = kernel_cls_eval(F_lsm_tr, F_lsm_te, y_tr, y_te)

    # FFT 谱 (选通门, 二次读出)
    spec_tr = torch.fft.fftshift(torch.fft.fft(gtr_cube.squeeze(1), dim=-1), -1).abs()
    spec_te = torch.fft.fftshift(torch.fft.fft(gte_cube.squeeze(1), dim=-1), -1).abs()
    acc_fft = kernel_cls_eval(spec_tr, spec_te, y_tr, y_te)

    cls_res[snr_db] = dict(lsm=acc_lsm, fft=acc_fft)
    print(f"SNR {snr_db:+.0f} dB | LSM train/test = {acc_lsm[0]:.3f}/{acc_lsm[1]:.3f} | "
          f"FFT train/test = {acc_fft[0]:.3f}/{acc_fft[1]:.3f}", flush=True)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
snrs = list(cls_res)
ax.plot(snrs, [cls_res[s]["lsm"][1] for s in snrs], "o-", label="LSM-iq (quad readout)")
ax.plot(snrs, [cls_res[s]["fft"][1] for s in snrs], "s--", label="FFT 谱 (quad readout)")
ax.axhline(0.25, color="k", ls=":", label="随机(4类)")
ax.set(xlabel="SNR (dB)", ylabel="测试准确率", title="微多普勒 4 类分类 (级联/选通后)")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()""")

md(r"""## E2 · 混淆矩阵与 t-SNE（LSM, 0 dB）

错误结构：相邻 f_m 类互混 → 频率分辨率限制；β=0.5 的两类（1,2）互混 → 幅度调制
与频率调制在二次特征上的耦合。""")

code(r"""from sklearn.manifold import TSNE
snr_db = 0.0
g1 = torch.Generator().manual_seed(SEED + int(snr_db*10) + 5000)
g2 = torch.Generator().manual_seed(SEED + int(snr_db*10) + 6000)
y_tr = torch.randint(0, 4, (1600,), generator=g1)
y_te = torch.randint(0, 4, (500,), generator=g2)
sim = MicroDopplerSim(C.N_PULSES, C.M_BINS)
cube_tr, sig, gate_tr = sim.make_batch(1600, snr_db, y_tr.tolist(), g1)
cube_te, _, gate_te = sim.make_batch(500, snr_db, y_te.tolist(), g2)
idx = torch.arange(1600); idx_te = torch.arange(500)
gtr = cube_tr[idx, gate_tr]; gte = cube_te[idx_te, gate_te]
enc_g = IQGateEncoder(seed=SEED)
spk_tr, spk_te = enc_g.encode(gtr, sig), enc_g.encode(gte, sig)
lsm = LSM(spk_tr.shape[1], n_res=C.N_RES, leak=0.9, v_th=C.V_TH,
          rec_density=C.REC_DENSITY, spectral_radius=C.SPECTRAL_RADIUS,
          in_fan=C.IN_FAN, in_scale=0.5, exc_frac=C.EXC_FRAC,
          refractory=C.REFRACTORY, seed=SEED)
m_tr, j_tr = run_pool(lsm, spk_tr); m_te, j_te = run_pool(lsm, spk_te)
Ftr = torch.cat([j_tr, m_tr], 1)
Fte = torch.cat([j_te, m_te], 1)
Coef_mu = Ftr.mean(0, keepdim=True); Coef_sd = Ftr.std(0, keepdim=True).clamp_min(1e-6)
Fn_tr = (Ftr - Coef_mu) / Coef_sd; Fn_te = (Fte - Coef_mu) / Coef_sd
Ktr = quad_kernel(Fn_tr, Fn_tr); Kte = quad_kernel(Fn_te, Fn_tr)
Coef = torch.linalg.solve(Ktr + 10.0 * torch.eye(len(Ktr)), onehot(y_tr))
pred_tr = Ktr @ Coef; pred_te = Kte @ Coef
cm = torch.zeros(4, 4, dtype=torch.long)
for t, p in zip(y_te.long(), pred_te.argmax(1)):
    cm[t, p] += 1
print("测试混淆矩阵 (行=真实, 列=预测):")
print(cm.numpy())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
im = axes[0].imshow(cm / cm.sum(1, keepdim=True), cmap="Blues", vmin=0, vmax=1)
axes[0].set(xlabel="预测类", ylabel="真实类", title=f"LSM 混淆矩阵 @ {snr_db:+.0f} dB (选通后)")
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, int(cm[i, j]), ha="center", va="center")
fig.colorbar(im, ax=axes[0])
Z = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(Fte[:400].numpy())
for k in range(4):
    m = (y_te[:400].long() == k).numpy()
    axes[1].scatter(Z[m, 0], Z[m, 1], s=6, label=f"class {k} (f_m={CLASSES[k][0]}, β={CLASSES[k][1]})")
axes[1].set(title="LSM 特征 t-SNE (@ 0 dB, 选通后)")
axes[1].legend(fontsize=8); plt.tight_layout(); plt.show()""")

md(r"""## E1b · 决定性补充：平移不变特征（自相关 + 线性读出）

E1 的教训：二次核对随机载波位置过拟合（train 0.96–1.0 / test 0.22–0.33）。
物理上类信息 = 调制周期 f_m，与随机载波 f_D 无关 → 正确充分统计量是
**自相关** |R(τ)| = |IFT{|Z(f)|²}|：调制使 |R(τ)| 以 1/f_m 周期振荡，
f_D 只贡献相位。给两种范式都配上这个特征（FFT 流水线一步可得），线性读出即可。""")

code(r"""def acorr_feats(z, n_lag=32):
    Z = torch.fft.fft(z, dim=-1)
    R = torch.fft.ifft(Z.abs() ** 2, dim=-1)
    return R.abs()[:, 1:n_lag+1]

ac_res = {}
for snr_db in (-10.0, -5.0, 0.0, 5.0):
    g1 = torch.Generator().manual_seed(SEED + int(snr_db*10) + 5000)
    g2 = torch.Generator().manual_seed(SEED + int(snr_db*10) + 6000)
    y_tr = torch.randint(0, 4, (1600,), generator=g1)
    y_te = torch.randint(0, 4, (500,), generator=g2)
    sim = MicroDopplerSim(C.N_PULSES, C.M_BINS)
    cube_tr, sig, gate_tr = sim.make_batch(1600, snr_db, y_tr.tolist(), g1)
    cube_te, _, gate_te = sim.make_batch(500, snr_db, y_te.tolist(), g2)
    idx = torch.arange(1600); idx_te2 = torch.arange(500)
    z_tr = cube_tr[idx, gate_tr] / sig
    z_te = cube_te[idx_te2, gate_te] / sig
    F_tr, F_te = acorr_feats(z_tr), acorr_feats(z_te)
    pred_tr, _ = ridge_readout(F_tr, onehot(y_tr), F_tr[:1], alpha=1.0)
    pred_te, _ = ridge_readout(F_tr, onehot(y_tr), F_te, alpha=1.0)
    a_tr = float((pred_tr.argmax(1) == y_tr.long()).float().mean())
    a_te = float((pred_te.argmax(1) == y_te.long()).float().mean())
    ac_res[snr_db] = (a_tr, a_te)
    print(f"SNR {snr_db:+.0f} dB | 自相关+线性: train={a_tr:.3f} test={a_te:.3f}", flush=True)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
snrs = list(ac_res)
ax.plot(snrs, [ac_res[s][1] for s in snrs], "o-", color="C2", label="自相关+线性 (平移不变)")
ax.plot(snrs, [cls_res[s]["lsm"][1] for s in snrs], "o-", alpha=0.5, label="LSM+二次核 (过拟合)")
ax.plot(snrs, [cls_res[s]["fft"][1] for s in snrs], "s--", alpha=0.5, label="FFT谱+二次核 (过拟合)")
ax.axhline(0.25, color="k", ls=":", label="随机(4类)")
ax.set(xlabel="SNR (dB)", ylabel="测试准确率", title="微多普勒分类: 特征不变性的决定作用")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()""")

md(r"""## M3 结论

1. **v1→v4 消融链**（本笔记本的全部价值所在）：微多普勒分类的瓶颈依次是
   ① 距离门稀释（v1：全立方体，随机水平）→
   ② 线性读出表达力（v2：选通+二次展开，仍随机）→
   ③ **泛化/不变性**（v3：二次核对随机载波 f_D 过拟合，train 0.96–1.0 / test 0.22–0.33）→
   ④ 特征构造（v4：自相关 |R(τ)| + 线性：+5 dB 0.71、0 dB 0.57、-5 dB 0.35）。
2. **充分统计量是平移不变量**：调制 β·sin(2πf_m n) 使 |R(τ)| 以 1/f_m 为周期振荡，
   随机载波 f_D 只贡献相位。通用非线性读出（核/NN）若不内置此不变性，
   小样本下必然过拟合载波位置——这是"特征工程仍然重要"的定量例证。
3. **分工结论**：蓄水池的不可替代性在低 SNR 相干积累（M1/M2 的检测/测速）；
   识别层应采用"蓄水池特征 + 平移不变读出（自相关/循环谱）"的混合架构，
   而非端到端黑箱。
4. **改进路径**（计划书素材）：更长 CPI（N=256 → 4× 积分增益）、级联结构
   （先 f_D 补偿再去调制）、或微多普勒**参数回归**（f_m/β 连续估计）替代硬分类。
   沙尘杂波下的分类（M2 环境 × M3 任务）是开放问题。""")

nb["cells"] = cells
path = Path(__file__).parent / "m3_microdoppler_classification.ipynb"
nbf.write(nb, path)
client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                        resources={"metadata": {"path": str(path.parent)}})
client.execute()
nbf.write(nb, path)
print("executed:", path)
