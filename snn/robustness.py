"""片上可行性: LSM 对权重量化与权重噪声的鲁棒性 (对应 QD-MLL PCU 的 ~6.74bit 精度)

对固定随机池的 W_in / W_rec 做:
  1) 均匀量化 (b bit, per-matrix max-abs 缩放)
  2) 加性高斯权重噪声 (相对标准差 sigma_w)
读出头在扰动后的池上重新训练 (读出是脊回归, 片上/离线都便宜),
评估检测 Pd 与多普勒 RMSE 的退化。
"""
import argparse
import torch

import config as C
from simulator import RadarEchoSimulator
from encoding import IQEncoder
from reservoir import LSM
from experiment import run_pool, eval_detection, eval_doppler, LSM_GRID


def quantize(W, bits):
    if bits is None or bits >= 32:
        return W.clone()
    s = W.abs().max().clamp_min(1e-9)
    q = torch.round(W / s * (2 ** (bits - 1) - 1)) / (2 ** (bits - 1) - 1)
    return q * s


def perturb(lsm, bits=None, sigma_w=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    lsm.W_in = quantize(lsm.W_in, bits)
    lsm.W_rec = quantize(lsm.W_rec, bits)
    if sigma_w > 0:
        lsm.W_in = lsm.W_in * (1 + sigma_w * torch.randn(lsm.W_in.shape, generator=g))
        lsm.W_rec = lsm.W_rec * (1 + sigma_w * torch.randn(lsm.W_rec.shape, generator=g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, nargs="+", default=[-10.0, 0.0])
    ap.add_argument("--n-train", type=int, default=1600)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--leak", type=float, default=0.9)
    ap.add_argument("--in-scale", type=float, default=0.5)
    args = ap.parse_args()
    torch.manual_seed(C.SEED)

    sim = RadarEchoSimulator(C.N_PULSES, C.M_BINS, C.PULSE_LEN, C.F_D_MAX, seed=C.SEED)
    enc = IQEncoder(clip=C.IQ_CLIP, max_rate=C.IQ_MAX_RATE, seed=C.SEED)

    for snr_db in args.snr:
        g = torch.Generator().manual_seed(C.SEED + int(snr_db * 10) + 1000)
        cube_tr, y_tr, fd_tr, sig_mf = sim.make_batch(args.n_train, snr_db, g)
        cube_te, y_te, fd_te, _ = sim.make_batch(args.n_test, snr_db, g)
        spk_tr = enc.encode(cube_tr, sig_mf)
        spk_te = enc.encode(cube_te, sig_mf)

        print(f"\n=== SNR {snr_db:+.0f} dB (leak={args.leak}, in_scale={args.in_scale}) ===")
        print(f"{'扰动':>18s} | {'Pd':>7s} | {'RMSE':>8s}")
        for tag, kw in [("理想 (无扰动)", dict(bits=None, sigma_w=0.0)),
                        ("量化 8 bit", dict(bits=8, sigma_w=0.0)),
                        ("量化 6 bit", dict(bits=6, sigma_w=0.0)),
                        ("量化 5 bit", dict(bits=5, sigma_w=0.0)),
                        ("量化 4 bit", dict(bits=4, sigma_w=0.0)),
                        ("量化 3 bit", dict(bits=3, sigma_w=0.0)),
                        ("权重噪声 1%", dict(bits=None, sigma_w=0.01)),
                        ("权重噪声 5%", dict(bits=None, sigma_w=0.05)),
                        ("权重噪声 10%", dict(bits=None, sigma_w=0.1))]:
            lsm = LSM(spk_tr.shape[1], n_res=C.N_RES, leak=args.leak, v_th=C.V_TH,
                      rec_density=C.REC_DENSITY, spectral_radius=C.SPECTRAL_RADIUS,
                      in_fan=C.IN_FAN, in_scale=args.in_scale, exc_frac=C.EXC_FRAC,
                      refractory=C.REFRACTORY, seed=C.SEED)
            perturb(lsm, seed=C.SEED + 7, **kw)
            m_tr, j_tr = run_pool(lsm, spk_tr)
            m_te, j_te = run_pool(lsm, spk_te)
            pd = eval_detection(m_tr, j_tr, m_te, j_te,
                                spk_tr.mean(-1), spk_te.mean(-1), y_tr, y_te)
            rmse = eval_doppler(m_tr, j_tr, m_te, j_te, y_tr, y_te, fd_tr, fd_te)
            print(f"{tag:>18s} | {pd:7.3f} | {rmse:8.4f}", flush=True)


if __name__ == "__main__":
    main()
