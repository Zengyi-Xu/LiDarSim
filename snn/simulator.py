"""相参脉冲串雷达回波仿真 + 匹配滤波 (脉冲压缩)

回波模型: 发射 LFM 相参脉冲串, 目标延迟 d 处产生回波,
慢时间 (脉冲间) 相位按 exp(j*2*pi*f_d*n) 进动 (多普勒)。
输出为匹配滤波 (脉压) 后的复数 "距离门 x 脉冲" 矩阵 (M_BINS, N_PULSES)。
"""
import numpy as np
import torch


class RadarEchoSimulator:
    def __init__(self, n_pulses, m_bins, pulse_len, f_d_max=0.4, seed=0):
        self.n_pulses = n_pulses
        self.m_bins = m_bins
        self.pulse_len = pulse_len
        self.f_d_max = f_d_max

        # LFM 线性调频脉冲 (单位模长), 瞬时频率 0 -> 0.25 (归一化), 无混叠
        k = torch.arange(pulse_len, dtype=torch.float64)
        self.chirp = torch.exp(1j * np.pi * 0.5 * k**2 / pulse_len).to(torch.complex64)

        # 匹配滤波器频响 H = conj(FFT(chirp, M)) -> 脉压增益 ||chirp||_2 = sqrt(K)
        self.H = torch.conj(torch.fft.fft(self.chirp, n=m_bins))

        # 目标延迟范围: 避开脉压主瓣拖尾与 CFAR 参考窗边缘
        self.delay_min = pulse_len + 4
        self.delay_max = m_bins - 24

    def make_batch(self, batch, snr_db, generator):
        """生成一批匹配滤波后的回波。

        参数
        ----
        batch:    样本数, 前半无目标 (纯噪声), 后半单目标 (Swerling I 起伏)
        snr_db:   匹配滤波输出峰值 SNR (单脉冲, 平均, dB) = A^2*K/sigma^2
        generator: torch.Generator (CPU)

        返回
        ----
        cube:       (batch, M_BINS, N_PULSES) complex64 匹配滤波输出
        has_target: (batch,) bool
        f_d:        (batch,) 归一化多普勒 (周期/脉冲)
        sigma_mf:   MF 输出噪声标准差 (标量, 编码门限用)
        """
        B = batch
        half = B // 2

        snr_lin = 10.0 ** (snr_db / 10.0)
        sigma = np.sqrt(self.pulse_len / snr_lin)    # 取平均幅度 A=1
        sigma_mf = float(sigma * np.sqrt(self.pulse_len))

        has_target = torch.zeros(B, dtype=torch.bool)
        has_target[half:] = True

        # 目标参数: 延迟 / 归一化多普勒 / 初相
        delay = torch.randint(self.delay_min, self.delay_max, (B,), generator=generator)
        f_d = (torch.rand(B, generator=generator) * 2 - 1) * self.f_d_max
        phase0 = torch.rand(B, generator=generator) * 2 * np.pi
        # Swerling I: 脉间恒定, 试验间瑞利起伏 -> A^2 服从均值 1 的指数分布
        U = torch.rand(B, generator=generator).clamp_min(1e-6)
        amp = torch.sqrt(-torch.log(U))

        # 脉压前信号: 在延迟处放加权 delta, 与 chirp 卷积
        n = torch.arange(self.n_pulses)
        ph = 2 * np.pi * f_d[:, None] * n[None, :] + phase0[:, None]
        echoes = torch.zeros(B, self.n_pulses, self.m_bins, dtype=torch.complex64)
        src = torch.where(has_target[:, None],
                          (amp[:, None] * torch.exp(1j * ph)).to(torch.complex64),
                          torch.zeros(B, self.n_pulses, dtype=torch.complex64))
        echoes[torch.arange(B), :, delay] = src
        sig = torch.fft.ifft(torch.fft.fft(echoes, dim=-1)
                             * torch.fft.fft(self.chirp, n=self.m_bins), dim=-1)

        # 复高斯白噪声
        noise = (sigma / np.sqrt(2.0)) * (
            torch.randn(B, self.n_pulses, self.m_bins, generator=generator)
            + 1j * torch.randn(B, self.n_pulses, self.m_bins, generator=generator))

        # 匹配滤波 (脉冲压缩), 输出 "距离门 x 脉冲"
        cube = torch.fft.ifft(torch.fft.fft(sig + noise, dim=-1) * self.H, dim=-1)
        cube = cube.transpose(1, 2).contiguous()      # (B, M, N)
        return cube, has_target, f_d, sigma_mf
