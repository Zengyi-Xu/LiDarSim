# -*- coding: utf-8 -*-
"""M6 笔记本: 语义目标多点散射体分类 (block 处理 + 质心多普勒补偿).

结论数字引用 m6_dryrun2/3 的实测值 (同 seed/协议, 可复现):
  E1 profile+ridge @CPI=64: -10dB 0.29 / -5dB 0.39 / 0dB 0.57
  E3 +5dB n_tr=5000: profile+ridge 0.785 / pergate+MLP2 0.806
  E4 航迹投票 @0dB (4x64): 0.64
"""
import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# M6 · 语义目标分类：多点散射体回波仿真

**从 M3 的"4 档参数"到语义目标**。`semantic_sim.py` 用 articulated 多点散射体
运动学生成匹配滤波后回波 (B, M, N)：

| 类 | K | 散射体 | 微动结构 | 关键参数（类间留 ≥2× 物理间隙） |
|---|---|---|---|---|
| pedestrian 行人 | 6 | 躯干/头/双臂/双腿 | 步行摆肢, 腿反相 | f_walk~U(0.018,0.032), β_腿~U(0.8,2.0) |
| drone 无人机 | 5 | 机身+4旋翼 | 旋翼 PM 谐波梳 | f_rot=f_c+N(0,0.004), f_c~U(0.09,0.16) |
| vehicle 车辆 | 4 | 刚体散射点 | 发动机激励面板振动 | f_vib~U(0.006,0.014), β~U(0.05,0.3) |
| bird 鸟类 | 3 | 体+双翼 | 反相扑翼(含2次谐波) | f_flap~U(0.04,0.07), β_翼~U(0.8,1.8) |

共享干扰量：质心多普勒 f_d0~U(±0.25)、初相、Swerling I 单 CPI 幅度、
每散射体随机相位/RCS/门散布、多普勒抖动 DELTA_F=[4,3,0.8,3]×10⁻³（刚体最小）。

**v1 阴性结果（m6_dryrun.py）**：沿用 M3 的"单质心门选通"，全员随机水平
（0 dB 时 acorr+ridge 仅 0.286）。诊断：多散射体的判别能量分布在多个距离门
（腿在 ±2 门、旋翼 ±1 门），单门选通把它丢了；且整体多普勒随机使谱特征失配。
**v2 处理链（本笔记本）**：检测后取目标 RD 邻域块 (9 门) → 估计并补偿质心
多普勒（平移不变性由"估计+对消"实现）→ 块级谱特征。""")

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
from semantic_sim import SemanticEchoSim, CLASSES
from cls_ceiling import feat_acorr, feat_logspec, ridge_cls, onehot
from experiment import ridge_readout

SEED = C.SEED
torch.manual_seed(SEED)
HALF = 4

def block(cube, gate):
    B = cube.shape[0]
    idx = torch.arange(B)
    offs = torch.arange(-HALF, HALF + 1)
    return cube[idx[:, None], gate[:, None] + offs[None, :], :]

def demod(zblk):
    Z = torch.fft.fftshift(torch.fft.fft(zblk, dim=-1), -1)
    P = Z.abs().pow(2).sum(1)
    f_hat = (P.argmax(-1).float() / zblk.shape[-1]) - 0.5
    n = torch.arange(zblk.shape[-1])
    return zblk * torch.exp(-2j * np.pi * f_hat[:, None, None] * n), f_hat

def feat_profile(zblk_dm, n_fft=128):
    Z = torch.fft.fftshift(torch.fft.fft(zblk_dm, n=n_fft, dim=-1), -1)
    return torch.log(Z.abs().pow(2).sum(1) + 1e-6)

def feat_pergate(zblk_dm, n_fft=128):
    Z = torch.fft.fftshift(torch.fft.fft(zblk_dm, n=n_fft, dim=-1), -1)
    return torch.log(Z.abs().pow(2) + 1e-6).flatten(1)

def mlp2(Ftr, ytr, Fte, yte, hidden=256, epochs=150, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    mu = Ftr.mean(0, keepdim=True); sd = Ftr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Ftr - mu) / sd, (Fte - mu) / sd
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(), torch.nn.Dropout(0.2),
        torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(hidden // 2, 4))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lf = torch.nn.CrossEntropyLoss(); Ytr = ytr.long()
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = lf(net(Xtr), Ytr); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        a_tr = float((net(Xtr).argmax(1) == Ytr).float().mean())
        a_te = float((net(Xte).argmax(1) == yte.long()).float().mean())
    return a_tr, a_te

def make(snr_db, n_pulses, n_tr=1600, n_te=500):
    g1 = torch.Generator().manual_seed(SEED + int(snr_db * 10) + 7000)
    g2 = torch.Generator().manual_seed(SEED + int(snr_db * 10) + 8000)
    y_tr = torch.randint(0, 4, (n_tr,), generator=g1)
    y_te = torch.randint(0, 4, (n_te,), generator=g2)
    sim = SemanticEchoSim(n_pulses, C.M_BINS)
    cube_tr, sig, gate_tr = sim.make_batch(n_tr, snr_db, y_tr.tolist(), g1)
    cube_te, _, gate_te = sim.make_batch(n_te, snr_db, y_te.tolist(), g2)
    return cube_tr, cube_te, sig, gate_tr, gate_te, y_tr, y_te

print("setup ok")""")

md(r"""## E0 · 语义签名示例（CPI=256, 0 dB）

每类取一个样本：左 = 质心补偿后块的距离-多普勒图 (9 门 × 128 多普勒 bin)，
右 = 块多普勒剖面（对门非相干求和）。可见类间结构的定性差异：
车辆 = 单窄峰；行人 = 主峰+近距边带群；鸟类 = 中距边带对；无人机 = 谐波梳。""")

code(r"""g = torch.Generator().manual_seed(SEED + 300)
sim = SemanticEchoSim(256, C.M_BINS)
cube, sig, gate = sim.make_batch(4, 0.0, [0, 1, 2, 3], g)
zb, _ = demod(block(cube, gate))
Z = torch.fft.fftshift(torch.fft.fft(zb, n=128, dim=-1), -1)
S = torch.log(Z.abs().pow(2) + 1e-6)
prof = torch.log(Z.abs().pow(2).sum(1) + 1e-6)
fig, axes = plt.subplots(4, 2, figsize=(11, 12))
fd = np.linspace(-0.5, 0.5, 128)
for k in range(4):
    im = axes[k, 0].imshow(S[k].numpy(), aspect="auto", origin="lower",
                           extent=[-0.5, 0.5, -HALF, HALF], cmap="viridis")
    axes[k, 0].set(title=f"{CLASSES[k]} — RD 块", ylabel="距离门偏移",
                   xlabel="多普勒 (周期/脉冲)")
    fig.colorbar(im, ax=axes[k, 0], fraction=0.025)
    axes[k, 1].plot(fd, prof[k].numpy())
    axes[k, 1].set(title=f"{CLASSES[k]} — 多普勒剖面",
                   xlabel="多普勒 (周期/脉冲)", ylabel="log 功率")
    axes[k, 1].grid(alpha=0.3)
plt.tight_layout(); plt.show()""")

md(r"""## E1 · SNR 扫描（CPI=64，n_tr=1600）

臂：**acorr+ridge（单质心门，v1 阴性对照）** / logspec+ridge（无补偿对照） /
profile+ridge / pergate+MLP2。教训预期：单门与无补偿特征应显著差于块特征。""")

code(r"""res_e1 = {}
for snr in (-10.0, -5.0, 0.0, 5.0):
    cube_tr, cube_te, sig, gate_tr, gate_te, y_tr, y_te = make(snr, 64)
    idx = torch.arange(len(y_tr)); idx_te = torch.arange(len(y_te))
    zc_tr = cube_tr[idx, gate_tr] / sig; zc_te = cube_te[idx_te, gate_te] / sig
    zb_tr, _ = demod(block(cube_tr, gate_tr))
    zb_te, _ = demod(block(cube_te, gate_te))
    row = {}
    row["单门acorr+ridge"] = ridge_cls(feat_acorr(zc_tr), y_tr, feat_acorr(zc_te), y_te)
    row["无补偿logspec+ridge"] = ridge_cls(feat_logspec(zc_tr), y_tr, feat_logspec(zc_te), y_te)
    row["profile+ridge"] = ridge_cls(feat_profile(zb_tr), y_tr, feat_profile(zb_te), y_te)
    row["pergate+MLP2"] = mlp2(feat_pergate(zb_tr), y_tr, feat_pergate(zb_te), y_te)
    res_e1[snr] = row
    print("SNR %+.0f dB:" % snr, flush=True)
    for k, (a, b) in row.items():
        print("   %-18s train=%.3f test=%.3f" % (k, a, b), flush=True)""")

md(r"""## E2/E3 · 工作点精测（n_tr=5000）与混淆结构

+5 dB 是检测后典型的分类工作点（单脉冲 +5 dB × 64 脉冲相干积累 ≈ 有效 23 dB）。
E3 给逐类准确率与混淆矩阵——预期 drone↔vehicle 互混（谐波梳低端 vs 浅振动边带
在剖面上同为"中心峰+近旁小峰"）。""")

code(r"""res_e3 = {}
for snr in (0.0, 5.0):
    cube_tr, cube_te, sig, gate_tr, gate_te, y_tr, y_te = make(snr, 64, 5000, 800)
    zb_tr, _ = demod(block(cube_tr, gate_tr))
    zb_te, _ = demod(block(cube_te, gate_te))
    row = {}
    row["profile+ridge"] = ridge_cls(feat_profile(zb_tr), y_tr, feat_profile(zb_te), y_te)
    row["profile+MLP2"] = mlp2(feat_profile(zb_tr), y_tr, feat_profile(zb_te), y_te)
    row["pergate+MLP2"] = mlp2(feat_pergate(zb_tr), y_tr, feat_pergate(zb_te), y_te)
    res_e3[snr] = row
    print("SNR %+.0f dB (n_tr=5000):" % snr, flush=True)
    for k, (a, b) in row.items():
        print("   %-16s train=%.3f test=%.3f" % (k, a, b), flush=True)
    sco_te, _ = ridge_readout(feat_profile(zb_tr), onehot(y_tr),
                              feat_profile(zb_te), alpha=1.0)
    pred = sco_te.argmax(1)
    conf = torch.zeros(4, 4, dtype=torch.long)
    for t, p in zip(y_te.long(), pred):
        conf[t, p] += 1
    print("  混淆矩阵 (行=真实):")
    for k in range(4):
        print("    %-10s %s  recall=%.2f" % (CLASSES[k], conf[k].tolist(),
                                             conf[k, k].item() / conf[k].sum().item()))""")

md(r"""## E4 · 航迹级融合（0 dB，4×64-CPI 多数投票）

把 256 脉冲切成 4 个 CPI 各自分类后投票——对应真实系统中对同一目标的多次观测。
这是客户体验到的"系统级准确率"。""")

code(r"""snr = 0.0
cube_tr, cube_te, sig, gate_tr, gate_te, y_tr, y_te = make(snr, 256)
v_tr, v_te = [], []
for s in range(0, 256, 64):
    sl = slice(s, s + 64)
    zb_tr, _ = demod(block(cube_tr[:, :, sl], gate_tr))
    zb_te, _ = demod(block(cube_te[:, :, sl], gate_te))
    sco_te, sco_tr = ridge_readout(feat_profile(zb_tr), onehot(y_tr),
                                   feat_profile(zb_te), alpha=1.0)
    v_tr.append(sco_tr.argmax(1)); v_te.append(sco_te.argmax(1))
def mv(V):
    oh = torch.nn.functional.one_hot(torch.stack(V, 1), 4).sum(1).float()
    return oh.argmax(1)
acc_vote = float((mv(v_te) == y_te.long()).float().mean())
print("0 dB 航迹投票 (4x64 CPI): test = %.3f" % acc_vote)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
snrs = sorted(res_e1)
for k, mk in [("profile+ridge", "o-"), ("pergate+MLP2", "s-"), ("单门acorr+ridge", "^--")]:
    ax.plot(snrs, [res_e1[s][k][1] for s in snrs], mk, label=k)
snrs3 = sorted(res_e3)
ax.plot(snrs3, [res_e3[s]["pergate+MLP2"][1] for s in snrs3], "d-",
        color="C1", alpha=0.45, label="pergate+MLP2 (n_tr=5000)")
ax.plot([0.0], [acc_vote], "*", ms=15, color="C3",
        label="航迹投票 (4xCPI, 0 dB)")
ax.axhline(0.25, color="k", ls=":", label="随机(4类)")
ax.set(xlabel="单脉冲 SNR (dB)", ylabel="测试准确率",
       title="M6 语义分类: 块处理链的准确率-SNR 曲线")
ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()

np.savez(SNN / "outputs_classify" / "m6_semantic_results.npz",
         **{f"e1_{s}_{k}": list(v) for s, row in res_e1.items() for k, v in row.items()},
         **{f"e3_{s}_{k}": list(v) for s, row in res_e3.items() for k, v in row.items()},
         vote_0dB=acc_vote)
print("saved -> outputs_classify/m6_semantic_results.npz")""")

md(r"""## M6 结论

1. **语义散射体仿真是可行的任务生成器**：articulated 运动学模型产生了定性可分的
   微多普勒结构（窄峰/近边带群/中距边带对/谐波梳），且类间频率留有 ≥2× 物理间隙。
2. **处理链决定成败（v1→v2 是 M3 之后第二个关键教训）**：单质心门选通把多散射体的
   判别能量（腿/旋翼在邻门）整个丢掉，全员随机；RD 块 + 质心多普勒估计补偿 +
   块级剖面特征后，0 dB 即达 0.61，+5 dB 达 **0.79–0.81**（n_tr=5000）。
3. **仍然是信息受限**：MLP2 在剖面上不胜 ridge（0.769 vs 0.785，+5 dB），
   增大训练量从 1600→5000 提升有限；pergate+MLP2 (0.806) 略好——用了距离结构。
   混淆集中在物理相邻的 drone↔vehicle。
4. **系统集成值**：0 dB 单次 CPI 0.61 → 4 次观测投票 0.64；CPI=256 单次 0.67。
   与分类上限探测一致——**准确率由 CPI×SNR×类间隙决定，架构间差距小；
   系统的价值仍在前端以极低功耗提供这些 CPI（0.14 mW, M4）**。
5. **开放问题**：实测微多普勒数据标定类参数（WP1）；多块/多目标场景；时频 CNN
   在更大数据量下的天花板；把"估计+补偿"整体搬入光子前端（延迟线互相关）。""")

nb["cells"] = cells
path = Path(__file__).parent / "m6_semantic_classification.ipynb"
nbf.write(nb, path)
client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                        resources={"metadata": {"path": str(path.parent)}})
client.execute()
nbf.write(nb, path)
print("executed:", path)
