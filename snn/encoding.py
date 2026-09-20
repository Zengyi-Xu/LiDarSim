"""匹配滤波输出 -> 脉冲序列 (spike train) 的编码方案

统一接口: encode(cube, sigma_mf) -> (B, C, N) float 0/1
    cube:    (B, M_BINS, N_PULSES) complex 匹配滤波输出
    sigma_mf: MF 输出噪声标准差, 作为门限/归一化单位
时间轴 = 脉冲序号 (慢时间), 即 SNN 的仿真步。
"""
import torch


class EventEncoder:
    """方案A: 门限检测事件化。幅值超门限的 (距离门, 脉冲) 位置发一个脉冲。
    通道 = 距离门数, 最稀疏、最事件驱动, 但门限以下信息丢失。"""

    name = "event"

    def __init__(self, thresh=4.0):
        self.thresh = thresh

    def encode(self, cube, sigma_mf):
        return (cube.abs() > self.thresh * sigma_mf).float()


class IQEncoder:
    """方案B: I/Q 群体编码。每个距离门 4 通道 (I+, I-, Q+, Q-),
    发放率正比于整流后的归一化幅值 (伯努利采样)。保留复数信息,
    弱信号下仍携带相位进动。rates() 输出连续发放率供 ESN 基线用。"""

    name = "iq"

    def __init__(self, clip=6.0, max_rate=0.8, seed=0):
        self.clip = clip
        self.max_rate = max_rate
        self.gen = torch.Generator().manual_seed(seed)

    def rates(self, cube, sigma_mf):
        z = cube / sigma_mf
        I, Q = z.real, z.imag
        r = torch.stack([I.clamp_min(0), (-I).clamp_min(0),
                         Q.clamp_min(0), (-Q).clamp_min(0)], dim=2)   # (B, M, 4, N)
        r = r.reshape(cube.shape[0], -1, cube.shape[-1])              # (B, 4M, N)
        return (r / self.clip).clamp(0, 1) * self.max_rate

    def encode(self, cube, sigma_mf):
        r = self.rates(cube, sigma_mf)
        return (torch.rand(r.shape, generator=self.gen) < r).float()


class DeltaEncoder:
    """方案C: 慢时间增量编码。相邻脉冲幅值变化超门限发 ON (上升) / OFF (下降)
    脉冲。背景平稳时极稀疏, 对突发目标敏感。"""

    name = "delta"

    def __init__(self, thresh=3.0):
        self.thresh = thresh

    def encode(self, cube, sigma_mf):
        a = cube.abs() / sigma_mf
        d = torch.diff(a, dim=-1, prepend=a[..., :1])
        on = (d > self.thresh).float()
        off = (-d > self.thresh).float()
        return torch.cat([on, off], dim=1)                            # (B, 2M, N)
