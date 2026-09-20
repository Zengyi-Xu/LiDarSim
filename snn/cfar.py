"""传统基线: MTD + 2D CA-CFAR 检测, FFT 多普勒估计, CRLB

CA-CFAR 门限不用解析因子: LFM 脉压噪声在距离维相关, 解析 Pfa 失真。
改为与 SNN 读出头同一协议 —— 在训练集噪声样本上按试验级 Pfa 标定门限。
"""
import numpy as np
import torch


def mtd_cfar_ratio(cube, tr_r=16, gd_r=3, tr_d=8, gd_d=2):
    """MTD (慢时间 FFT) 后计算二维 CA-CFAR 比值图 CUT/(训练环均值)。

    cube: (B, M, N) complex
    返回 ratio: (B, M, N), valid: (M, N) —— 仅参考窗完整的单元参与判决。
    """
    B, M, N = cube.shape
    rd = torch.fft.fftshift(torch.fft.fft(cube, dim=-1), dim=-1).abs() ** 2
    x = rd.unsqueeze(1)                                       # (B,1,M,N)

    def box_sum(kh, kw):
        w = torch.ones(1, 1, kh, kw)
        return torch.nn.functional.conv2d(x, w, padding=(kh // 2, kw // 2)).squeeze(1)

    Kh, Kw = 2 * (tr_r + gd_r) + 1, 2 * (tr_d + gd_d) + 1     # 参考窗 (大)
    kh, kw = 2 * gd_r + 1, 2 * gd_d + 1                       # 保护窗 (小)
    ring = box_sum(Kh, Kw) - box_sum(kh, kw)                  # 训练环之和
    L = Kh * Kw - kh * kw
    ratio = rd * L / ring.clamp_min(1e-9)

    valid = torch.zeros(M, N, dtype=torch.bool)
    valid[Kh // 2: M - Kh // 2, Kw // 2: N - Kw // 2] = True
    return ratio, valid


def calibrate_threshold(ratio_noise, valid, p_fa, iters=40):
    """在训练集噪声样本上按试验级 Pfa 二元搜索门限。"""
    hit = lambda t: ((ratio_noise > t) & valid).flatten(1).any(1).float().mean()
    lo, hi = 0.0, 1e5
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if hit(mid) > p_fa:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mtd_cfar_detect(ratio, valid, thr):
    return ((ratio > thr) & valid).flatten(1).any(dim=1)


def fft_doppler_est(cube):
    """最强距离门的慢时间 FFT + 抛物线插值, 估计归一化多普勒 (周期/脉冲)。"""
    B, M, N = cube.shape
    rd = torch.fft.fftshift(torch.fft.fft(cube, dim=-1), dim=-1).abs() ** 2
    r_star = rd.sum(dim=-1).argmax(dim=1)
    spec = rd[torch.arange(B), r_star]
    k0 = spec.argmax(dim=1).clamp(1, N - 2)
    b = torch.arange(B)
    y0, y1, y2 = spec[b, k0 - 1], spec[b, k0], spec[b, k0 + 1]
    denom = (y0 - 2 * y1 + y2).clamp_min(1e-12)
    delta = (0.5 * (y0 - y2) / denom).clamp(-1, 1)
    return (k0 + delta - N / 2) / N


def crlb_doppler(snr_peak_lin, n):
    """复指数频率估计的 CRLB 标准差, 单位: 周期/脉冲。

    snr_peak_lin: 单脉冲 MF 输出峰值 SNR (线性)
    """
    return float(np.sqrt(6.0 / ((2 * np.pi) ** 2 * snr_peak_lin * n * (n ** 2 - 1))))
