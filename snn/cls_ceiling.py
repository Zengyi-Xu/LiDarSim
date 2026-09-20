# -*- coding: utf-8 -*-
"""分类准确率上限探测 (回应: 器件工作点上的分类到底能到多少).

M3 用弱读出(线性/二次核)隔离"平移不变性"变量, 得到的 0.57/0.71 是下限.
本脚本在相同 4 类微多普勒任务上比较:
  1. acorr + ridge          -- M3 v4 参考线
  2. log 谱 + ridge         -- 经典周期图特征 (载波 f_D 使其失配)
  3. 倒谱 + ridge           -- 周期估计的专业特征
  4. acorr + MLP            -- 不变特征 + 学习读出
  5. LSM 池状态 + MLP       -- SNN 范式内部: 学习读出能否自己解出 f_m vs f_D
  6. oracle 去载波 + log谱 + ridge -- 已知 f_D 完美对消的上限参照
附加: CPI=256 (4倍积累) 下重测最佳管道, 看时间资源换准确率.
"""
import os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config as C
from experiment import run_pool, ridge_readout
from reservoir import LSM

SEED = C.SEED
torch.manual_seed(SEED)
CLASSES = [(0.02, 0.2), (0.05, 0.5), (0.10, 0.5), (0.20, 0.2)]
OUT = HERE / "outputs_classify"
OUT.mkdir(exist_ok=True)


class MicroDopplerSim:
    """匹配滤波后 (B, M, N); 返回真实门与 f_D (oracle 用)"""
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
            ph = (2 * np.pi * f_d[b] * n + beta * np.sin(2 * np.pi * fm * n) + phi0[b])
            y[b, delay[b], :] = (amp[b] * torch.exp(1j * ph)).to(torch.complex64)
        noise = (sigma_mf / np.sqrt(2)) * (
            torch.randn(B, self.M, self.N, generator=generator)
            + 1j * torch.randn(B, self.M, self.N, generator=generator))
        return y + noise, sigma_mf, delay, f_d


class IQGateEncoder:
    """选通门慢时间序列 (B, N) -> (B, 4, N) 群体发放率编码 (M3 同款)"""
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


# ---------------- 特征 ----------------
def feat_acorr(z, L=32):
    Z = torch.fft.fft(z, dim=-1)
    R = torch.fft.ifft(Z.abs() ** 2, dim=-1)
    return R.abs()[:, 1:L + 1]


def feat_logspec(z):
    Z = torch.fft.fftshift(torch.fft.fft(z, dim=-1), -1)
    return torch.log(Z.abs() ** 2 + 1e-6)


def feat_cepstrum(z, L=32):
    Z = torch.fft.fft(z, dim=-1)
    P = torch.log(Z.abs() ** 2 + 1e-6)
    return torch.fft.ifft(P, dim=-1).real[:, 1:L + 1]


def feat_oracle_logspec(z, f_d):
    n = torch.arange(z.shape[-1])
    zd = z * torch.exp(-2j * np.pi * f_d[:, None] * n)
    return feat_logspec(zd)


# ---------------- 读出 ----------------
def ridge_cls(Ftr, ytr, Fte, yte, alpha=1.0):
    sco_te, sco_tr = ridge_readout(Ftr, onehot(ytr), Fte, alpha=alpha)
    a_te = float((sco_te.argmax(1) == yte.long()).float().mean())
    a_tr = float((sco_tr.argmax(1) == ytr.long()).float().mean())
    return a_tr, a_te


def mlp_cls(Ftr, ytr, Fte, yte, hidden=128, epochs=80, lr=1e-3, seed=0):
    """标准化 + 小 MLP (学习不变性/非线性边界的能力上限探测器)"""
    torch.manual_seed(seed)
    mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Ftr - mu) / sd, (Fte - mu) / sd
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(hidden, 4))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lf = torch.nn.CrossEntropyLoss()
    Ytr = ytr.long()
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = lf(net(Xtr), Ytr); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        a_tr = float((net(Xtr).argmax(1) == Ytr).float().mean())
        a_te = float((net(Xte).argmax(1) == yte.long()).float().mean())
    return a_tr, a_te


def lsm_mlp(z, sig, ytr, yte, seed=SEED):
    """SNN 范式: IQ 群体编码 -> 256 池 LSM -> [轨迹快照+均值] -> MLP"""
    enc = IQGateEncoder(seed=seed)
    spk_tr, spk_te = enc.encode(z[0], sig), enc.encode(z[1], sig)
    lsm = LSM(spk_tr.shape[1], n_res=C.N_RES, leak=0.9, v_th=C.V_TH,
              rec_density=C.REC_DENSITY, spectral_radius=C.SPECTRAL_RADIUS,
              in_fan=C.IN_FAN, in_scale=0.5, exc_frac=C.EXC_FRAC,
              refractory=C.REFRACTORY, seed=seed)
    m_tr, j_tr = run_pool(lsm, spk_tr)
    m_te, j_te = run_pool(lsm, spk_te)
    Ftr = torch.cat([j_tr, m_tr], 1)
    Fte = torch.cat([j_te, m_te], 1)
    return mlp_cls(Ftr, ytr, Fte, yte, seed=seed)


# ---------------- 主实验 ----------------
def run(snr_db, n_pulses=64, n_tr=1600, n_te=500):
    g1 = torch.Generator().manual_seed(SEED + int(snr_db * 10) + 5000)
    g2 = torch.Generator().manual_seed(SEED + int(snr_db * 10) + 6000)
    y_tr = torch.randint(0, 4, (n_tr,), generator=g1)
    y_te = torch.randint(0, 4, (n_te,), generator=g2)
    sim = MicroDopplerSim(n_pulses, C.M_BINS)
    cube_tr, sig, gate_tr, fd_tr = sim.make_batch(n_tr, snr_db, y_tr.tolist(), g1)
    cube_te, _, gate_te, fd_te = sim.make_batch(n_te, snr_db, y_te.tolist(), g2)
    idx = torch.arange(n_tr); idx_te = torch.arange(n_te)
    z_tr = cube_tr[idx, gate_tr] / sig
    z_te = cube_te[idx_te, gate_te] / sig
    row = {}
    row["acorr+ridge"] = ridge_cls(feat_acorr(z_tr), y_tr, feat_acorr(z_te), y_te)
    row["logspec+ridge"] = ridge_cls(feat_logspec(z_tr), y_tr, feat_logspec(z_te), y_te)
    row["ceps+ridge"] = ridge_cls(feat_cepstrum(z_tr), y_tr, feat_cepstrum(z_te), y_te)
    row["acorr+MLP"] = mlp_cls(feat_acorr(z_tr), y_tr, feat_acorr(z_te), y_te)
    row["oracle+ridge"] = ridge_cls(feat_oracle_logspec(z_tr, fd_tr), y_tr,
                                    feat_oracle_logspec(z_te, fd_te), y_te)
    row["LSM+MLP"] = lsm_mlp((z_tr, z_te), sig, y_tr, y_te)
    return row


def main():
    t0 = time.time()
    results = {}
    print("%-6s | %-22s %-22s" % ("SNR", "方法 (train/test)", ""), flush=True)
    for snr_db in (-10.0, -5.0, 0.0, 5.0):
        row = run(snr_db)
        results[snr_db] = row
        print("SNR %+.0f dB (CPI=64):" % snr_db, flush=True)
        for k, (a_tr, a_te) in row.items():
            print("   %-16s train=%.3f  test=%.3f" % (k, a_tr, a_te), flush=True)
    # CPI=256: 时间资源换准确率
    print("\nCPI=256 (4x 积累), 0 dB:", flush=True)
    row256 = run(0.0, n_pulses=256)
    for k, (a_tr, a_te) in row256.items():
        print("   %-16s train=%.3f  test=%.3f" % (k, a_tr, a_te), flush=True)
    results["cpi256_0dB"] = row256

    flat = {}
    for snr, row in results.items():
        for k, (a_tr, a_te) in row.items():
            flat[f"{snr}_{k}"] = [a_tr, a_te]
    np.savez(OUT / "cls_ceiling_results.npz", **flat)
    print("\nsaved -> outputs_classify/cls_ceiling_results.npz  (%.1f s)" % (time.time() - t0))


if __name__ == "__main__":
    main()
