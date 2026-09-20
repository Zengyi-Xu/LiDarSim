"""蓄水池网络: LSM (脉冲) 与 ESN (连续值基线)

两者共同特点: 池连接随机且固定, 只训练线性读出头 (脊回归),
训练是凸优化, 秒级收敛, 可在线学习 —— 这是蓄水池方案的效率来源。

run() 返回两种特征:
    mean: (B, n_res)  时间平均的低通态 (慢分量/能量)
    traj: (B, K*n_res) 低通态按时间下采样的轨迹快照 (保留相位进动信息)
"""
import numpy as np
import torch

TRAJ_K = 16          # 轨迹快照数
READOUT_BETA = 0.9   # 读出头低通常数 (记忆 ~10 脉冲)


class LSM:
    """Liquid State Machine: 随机稀疏 LIF 神经元池 (Dale 定律 + 不应期)。"""

    def __init__(self, n_in, n_res=256, leak=0.55, v_th=1.0,
                 rec_density=0.12, spectral_radius=0.95,
                 in_fan=32, in_scale=0.3, exc_frac=0.8, refractory=2, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.n_in, self.n_res, self.leak, self.v_th = n_in, n_res, leak, v_th
        self.refractory = refractory

        # 输入权重: 每个神经元随机接 in_fan 个输入通道, 符号随机
        rows = torch.arange(n_res).repeat_interleave(in_fan)
        cols = torch.randint(0, n_in, (n_res * in_fan,), generator=g)
        vals = (torch.randint(0, 2, (n_res * in_fan,), generator=g) * 2 - 1).float()
        self.W_in = torch.zeros(n_res, n_in)
        self.W_in[rows, cols] = vals * in_scale

        # 递归权重: 稀疏随机, 源神经元兴奋/抑制固定 (Dale), 抑制强 2 倍
        mask = torch.rand(n_res, n_res, generator=g) < rec_density
        W = torch.randn(n_res, n_res, generator=g) * mask
        exc = torch.rand(n_res, generator=g) < exc_frac
        sign = torch.where(exc, 1.0, -2.0)
        W = W * sign[:, None]
        lam = torch.linalg.eigvals(W).abs().max()
        self.W_rec = W * (spectral_radius / lam)
        self.last_spike_rate = 0.0      # 池内平均发放率 (能耗指标)

    @torch.no_grad()
    def run(self, spikes, collect=False):
        B, C, T = spikes.shape
        v = torch.zeros(B, self.n_res)
        r = torch.zeros(B, self.n_res)
        s_prev = torch.zeros(B, self.n_res)
        refrac = torch.zeros(B, self.n_res)
        total = 0.0
        snaps = []
        trace = []
        for t in range(T):
            drive = spikes[:, :, t] @ self.W_in.T + s_prev @ self.W_rec.T
            v = torch.where(refrac > 0, torch.zeros_like(v), self.leak * v + drive)
            s = (v >= self.v_th).float()
            v = v - s * self.v_th          # 减法式复位
            refrac = torch.where(s > 0, torch.full_like(refrac, float(self.refractory)),
                                 (refrac - 1).clamp_min(0))
            r = READOUT_BETA * r + s
            s_prev = s
            total += float(s.sum())
            if t % max(1, T // TRAJ_K) == 0 or t == T - 1:
                snaps.append(r.clone())
            if collect:
                trace.append(s)
        self.last_spike_rate = total / (B * T * self.n_res)
        mean = r / T
        traj = torch.cat(snaps[:TRAJ_K], dim=1)          # (B, K*n_res)
        if collect:
            return mean, traj, torch.stack(trace, dim=-1)
        return mean, traj


class ESN:
    """回声状态网络基线: 连续值输入 (发放率), 泄漏 tanh 池。
    与 LSM 同为随机固定池 + 线性读出头, 用于消融 "脉冲化" 的增益。"""

    def __init__(self, n_in, n_res=256, leak=0.2, in_scale=1.0,
                 rec_density=0.12, spectral_radius=0.95, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.n_in, self.n_res, self.leak = n_in, n_res, leak
        self.W_in = torch.randn(n_res, n_in, generator=g) * (in_scale / np.sqrt(n_in))
        mask = torch.rand(n_res, n_res, generator=g) < rec_density
        W = torch.randn(n_res, n_res, generator=g) * mask
        lam = torch.linalg.eigvals(W).abs().max()
        self.W_rec = W * (spectral_radius / lam)

    @torch.no_grad()
    def run(self, u):
        B, C, T = u.shape
        h = torch.zeros(B, self.n_res)
        r = torch.zeros(B, self.n_res)
        snaps = []
        for t in range(T):
            h = (1 - self.leak) * h + self.leak * torch.tanh(
                u[:, :, t] @ self.W_in.T + h @ self.W_rec.T)
            r = READOUT_BETA * r + h
            if t % max(1, T // TRAJ_K) == 0 or t == T - 1:
                snaps.append(r.clone())
        mean = r / T
        traj = torch.cat(snaps[:TRAJ_K], dim=1)
        return mean, traj
