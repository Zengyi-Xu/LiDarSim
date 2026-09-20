# -*- coding: utf-8 -*-
"""M2 实验库 + 笔记本构建: 非高斯杂波环境下的检测。
沙尘: K 分布复合杂波 (spiky, shape ν 小); 大雾: 高 CNR 高斯杂波 (ν→∞)。
"""
import numpy as np
import torch


class ClutterSimulator:
    """在 RadarEchoSimulator 基础上加 K 分布杂波背景。

    K 杂波 = 快变复高斯 (speckle) × 慢变 Gamma 纹理 (texture, 空间相关)。
    CNR: 杂波与热噪声比; nu: Gamma 形状 (nu→∞ 退化为高斯, nu 小=尖峰状沙尘)。
    """

    def __init__(self, base_sim, cnr_db=10.0, nu=2.0, tex_corr_bins=6.0, seed=0):
        self.base = base_sim
        self.cnr_db = cnr_db
        self.nu = nu
        self.tex_corr = tex_corr_bins
        self.g = torch.Generator().manual_seed(seed + 77)

    def make_batch(self, batch, snr_db):
        cube, has_target, f_d, sigma_mf = self.base.make_batch(batch, snr_db, self.g)
        B, M, N = cube.shape
        c_lin = 10.0 ** (self.cnr_db / 10.0)
        if self.nu > 50:
            tex = torch.ones(B, M, 1)
        else:
            # Gamma 纹理: 空间相关 (range 维高斯滤波), 形状 nu, 均值 1
            z = torch.randn(B, M, generator=self.g)
            k = int(4 * self.tex_corr)
            ker = torch.exp(-0.5 * (torch.arange(-k, k + 1) / self.tex_corr) ** 2)
            ker = ker / ker.sum()
            zc = torch.nn.functional.conv1d(z.unsqueeze(1), ker.view(1, 1, -1),
                                            padding=k).squeeze(1)
            zc = (zc - zc.mean(dim=1, keepdim=True)) / zc.std(dim=1, keepdim=True).clamp_min(1e-6)
            tex = (1 + zc / np.sqrt(self.nu)).clamp_min(0.05).unsqueeze(-1)  # 均值1, 方差1/nu
        # 杂波 = 纹理 × 复高散斑, 总杂波功率 = c_lin × 热噪声
        sigma = np.sqrt(self.base.pulse_len / 10.0 ** (snr_db / 10))
        sig_c = sigma * np.sqrt(c_lin)
        clutter = (sig_c / np.sqrt(2)) * torch.sqrt(tex) * (
            torch.randn(B, M, N, generator=self.g)
            + 1j * torch.randn(B, M, N, generator=self.g))
        return cube + clutter, has_target, f_d, sigma_mf


def measured_pfa(cfar_detect_fn, cube_noise):
    """噪声样本上实测虚警率 (CFAR 失配度量)。"""
    return float(cfar_detect_fn(cube_noise).float().mean())
