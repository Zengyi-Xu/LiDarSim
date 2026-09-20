# -*- coding: utf-8 -*-
"""语义目标多点散射体回波仿真器 (M6).

把 M3 的单散射体+单正弦调制升级为 articulated 多散射体运动学模型:
类别判据从"4 档 (f_m, β) 参数"变为语义目标类型, 每类由多个点散射体的
相对运动 (肢体摆动 / 旋翼谐波 / 面板振动 / 扑翼) 产生丰富微多普勒结构.

输出与 MicroDopplerSim 同构: 匹配滤波后复数立方体 (B, M, N), 噪声 sigma,
质心门 (供 oracle 选通). 全部向量化的 torch 实现.

类别运动学 (归一化单位: 频率=周期/脉冲, β=弧度):

| 类 | K | 散射体 | 微动来源 | 关键参数 |
|---|---|---|---|---|
| 0 pedestrian 行人 | 6 | 躯干/头/双臂/双腿 | 步行摆肢, 腿反相 | f_walk~U(0.018,0.032), β_腿~U(0.8,2.0) |
| 1 drone 无人机 | 5 | 机身+4旋翼 | 旋翼 PM 谐波梳 (blade flash) | f_rot=f_c+N(0,0.004), f_c~U(0.09,0.16), β1~U(0.6,1.2) |
| 2 vehicle 车辆 | 4 | 刚体散射点 | 发动机激励面板振动 | f_vib~U(0.006,0.014), β~U(0.05,0.3) |
| 3 bird 鸟类 | 3 | 体+双翼 | 扑翼反相 (含 2 次谐波) | f_flap~U(0.04,0.07), β_翼~U(0.8,1.8) |

类间调制频率保留 >=2x 物理间隙 (发动机振动 << 步行摆肢 << 扑翼 << 旋翼),
类内仍保留实现随机性; 旋翼同型电机转速相近 -> 共享 f_c 仅小抖动.
每散射体多普勒抖动按类设定 (刚体最小): DELTA_F = [0.004, 0.003, 0.0008, 0.003].

所有类共享随机干扰量: 质心多普勒 f_d0~U(-f_d_max,f_d_max), 初相,
Swerling I 单 CPI 幅度起伏, 每散射体 RCS 权重与距离门散布.
"""
import numpy as np
import torch

CLASSES = ["pedestrian", "drone", "vehicle", "bird"]


class SemanticEchoSim:
    def __init__(self, n_pulses, m_bins, f_d_max=0.25):
        self.N, self.M = n_pulses, m_bins
        self.f_d_max = f_d_max

    @staticmethod
    def _term(beta, f, psi, n):
        """(nc,T) 参数 -> (nc,T,N) 正弦 PM 项"""
        return beta[:, :, None] * torch.sin(
            2 * np.pi * f[:, :, None] * n[None, None, :] + psi[:, :, None])

    def _kinematics(self, c, nc, n, gen):
        """返回 (nc, Kc, N) 每散射体 PM 相位, 门偏移 (Kc,), RCS 权重 (Kc,)"""
        U = lambda a, b, *s: a + (b - a) * torch.rand(*s, generator=gen)
        Z = lambda *s: torch.zeros(*s)
        PI = np.pi
        if c == 0:  # pedestrian: 躯干/头 (2f_walk 小幅), 臂 (反相), 腿 (反相)
            f_w = U(0.018, 0.032, nc, 1)
            M = torch.cat([
                self._term(U(0.05, 0.12, nc, 1), 2 * f_w, Z(nc, 1), n),
                self._term(U(0.05, 0.12, nc, 1), 2 * f_w, Z(nc, 1), n),
                self._term(U(0.5, 1.2, nc, 1), f_w, torch.full((nc, 1), PI / 2), n),
                self._term(U(0.5, 1.2, nc, 1), f_w, torch.full((nc, 1), 3 * PI / 2), n),
                self._term(U(0.8, 2.0, nc, 1), f_w, Z(nc, 1), n),
                self._term(U(0.8, 2.0, nc, 1), f_w, torch.full((nc, 1), PI), n)], 1)
            off = torch.tensor([0, 0, -1, 1, -2, 2])
            w = torch.tensor([1.0, 0.5, 0.45, 0.45, 0.7, 0.7])
        elif c == 1:  # drone: 机身微晃 + 4 旋翼各 3 次谐波梳 (同型电机共享 f_c)
            f_c = U(0.09, 0.16, nc, 1)
            f_rot = f_c + 0.004 * torch.randn(nc, 4, generator=gen)
            b1 = U(0.6, 1.2, nc, 4)
            rotors = [sum(self._term(b1[:, k:k + 1] * (0.55 ** (h - 1)),
                                     h * f_rot[:, k:k + 1],
                                     2 * PI * torch.rand(nc, 1, generator=gen), n)
                            for h in (1, 2, 3)) for k in range(4)]
            body = self._term(U(0.02, 0.06, nc, 1), U(0.01, 0.03, nc, 1),
                              2 * PI * torch.rand(nc, 1, generator=gen), n)
            M = torch.cat([body] + rotors, 1)
            off = torch.tensor([0, -1, 1, -1, 1])
            w = torch.tensor([1.0, 0.35, 0.35, 0.35, 0.35])
        elif c == 2:  # vehicle: 刚体散射点各带浅振动
            M = self._term(U(0.05, 0.30, nc, 4), U(0.006, 0.014, nc, 4),
                           2 * PI * torch.rand(nc, 4, generator=gen), n)
            off = torch.tensor([-1, 0, 0, 1])
            w = torch.tensor([1.0, 0.7, 0.45, 0.35])
        else:         # bird: 体微动 + 双翼反相扑翼 (含 2 次谐波)
            f_f = U(0.04, 0.07, nc, 1)
            bw = U(0.8, 1.8, nc, 1)
            wing_l = (self._term(bw, f_f, Z(nc, 1), n)
                      + self._term(0.3 * bw, 2 * f_f, torch.full((nc, 1), PI), n))
            wing_r = (self._term(bw, f_f, torch.full((nc, 1), PI), n)
                      + self._term(0.3 * bw, 2 * f_f, Z(nc, 1), n))
            body = self._term(U(0.03, 0.10, nc, 1), f_f, Z(nc, 1), n)
            M = torch.cat([body, wing_l, wing_r], 1)
            off = torch.tensor([0, -1, 1])
            w = torch.tensor([1.0, 0.6, 0.6])
        return M, off, w

    def make_batch(self, batch, snr_db, cls_list, generator):
        B = batch
        sigma_mf = float(np.sqrt(1.0 / 10.0 ** (snr_db / 10.0)))
        gate = torch.randint(12, self.M - 12, (B,), generator=generator)
        f_d0 = (torch.rand(B, generator=generator) * 2 - 1) * self.f_d_max
        phi0 = torch.rand(B, generator=generator) * 2 * np.pi
        U = torch.rand(B, generator=generator).clamp_min(1e-6)
        amp = torch.sqrt(-torch.log(U))
        n = torch.arange(self.N, dtype=torch.float32)
        y = torch.zeros(B, self.M, self.N, dtype=torch.complex64)
        cls = torch.as_tensor(cls_list)
        DELTA_F = [0.004, 0.003, 0.0008, 0.003]  # 每散射体多普勒抖动 (刚体最小)
        for c in range(4):
            idx = (cls == c).nonzero(as_tuple=True)[0]
            nc = len(idx)
            if nc == 0:
                continue
            M, off, w = self._kinematics(c, nc, n, generator)
            Kc = off.shape[0]
            fd_k = f_d0[idx][:, None] + DELTA_F[c] * torch.randn(nc, Kc, generator=generator)
            ph = (2 * np.pi * fd_k[:, :, None] * n[None, None, :]
                  + M + phi0[idx][:, None, None]
                  + 2 * np.pi * torch.rand(nc, Kc, 1, generator=generator))
            sig_k = (amp[idx][:, None] * w[None, :].expand(nc, Kc))[..., None] \
                * torch.exp(1j * ph)
            for k in range(Kc):
                y[idx, gate[idx] + off[k], :] += sig_k[:, k, :].to(torch.complex64)
        noise = (sigma_mf / np.sqrt(2)) * (
            torch.randn(B, self.M, self.N, generator=generator)
            + 1j * torch.randn(B, self.M, self.N, generator=generator))
        return y + noise, sigma_mf, gate


if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)
    sim = SemanticEchoSim(64, 128)
    y, sig, gate = sim.make_batch(8, 0.0, [0, 1, 2, 3, 0, 1, 2, 3], g)
    e = y.abs().pow(2).sum(-1)
    print("signal check: peak gate match =",
          float((e.argmax(-1) == gate).float().mean()),
          "| cube", tuple(y.shape), "| sig=%.3f" % sig)
