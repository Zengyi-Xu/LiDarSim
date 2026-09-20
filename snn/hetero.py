"""异构蓄水池: 每个神经元独立 leak (模拟微环阵列 Q 值天然不一致)。
对比同质 LSM, 看高 SNR 平台期是否抬升 (回答: 需要更多非线性还是更多结构?)
"""
import argparse
import torch

import config as C
from simulator import RadarEchoSimulator
from encoding import IQEncoder
from experiment import run_pool, eval_detection, eval_doppler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, nargs="+", default=[-10.0, 0.0, 5.0])
    ap.add_argument("--n-train", type=int, default=1600)
    ap.add_argument("--n-test", type=int, default=500)
    args = ap.parse_args()
    torch.manual_seed(C.SEED)

    from reservoir import LSM

    class Hetero(LSM):
        def __init__(self, n_in, leak_lo=0.6, leak_hi=0.99, **kw):
            super().__init__(n_in, **kw)
            g = torch.Generator().manual_seed(kw.get("seed", 0) + 1)
            u = torch.rand(self.n_res, generator=g)
            import math
            self.leak_vec = torch.exp(u * (math.log(leak_hi) - math.log(leak_lo)) + math.log(leak_lo))

        @torch.no_grad()
        def run(self, spikes, collect=False):
            B, C, T = spikes.shape
            v = torch.zeros(B, self.n_res)
            r = torch.zeros(B, self.n_res)
            s_prev = torch.zeros(B, self.n_res)
            refrac = torch.zeros(B, self.n_res)
            total = 0.0
            snaps = []
            for t in range(T):
                drive = spikes[:, :, t] @ self.W_in.T + s_prev @ self.W_rec.T
                v = torch.where(refrac > 0, torch.zeros_like(v), self.leak_vec * v + drive)
                s = (v >= self.v_th).float()
                v = v - s * self.v_th
                refrac = torch.where(s > 0, torch.full_like(refrac, float(self.refractory)),
                                     (refrac - 1).clamp_min(0))
                r = 0.9 * r + s
                s_prev = s
                total += float(s.sum())
                if t % max(1, T // 16) == 0 or t == T - 1:
                    snaps.append(r.clone())
            self.last_spike_rate = total / (B * T * self.n_res)
            return r / T, torch.cat(snaps[:16], dim=1)

    sim = RadarEchoSimulator(C.N_PULSES, C.M_BINS, C.PULSE_LEN, C.F_D_MAX, seed=C.SEED)
    enc = IQEncoder(clip=C.IQ_CLIP, max_rate=C.IQ_MAX_RATE, seed=C.SEED)

    for snr_db in args.snr:
        g = torch.Generator().manual_seed(C.SEED + int(snr_db * 10) + 1000)
        cube_tr, y_tr, fd_tr, sig_mf = sim.make_batch(args.n_train, snr_db, g)
        cube_te, y_te, fd_te, _ = sim.make_batch(args.n_test, snr_db, g)
        spk_tr, spk_te = enc.encode(cube_tr, sig_mf), enc.encode(cube_te, sig_mf)
        print(f"\n=== SNR {snr_db:+.0f} dB ===")
        for tag, cls, kw in [
            ("同质 leak=0.9", LSM, dict(leak=0.9, in_scale=0.5)),
            ("异构 leak~[0.6,0.99]", Hetero, dict(in_scale=0.5)),
        ]:
            lsm = cls(spk_tr.shape[1], n_res=C.N_RES, v_th=C.V_TH,
                      rec_density=C.REC_DENSITY, spectral_radius=C.SPECTRAL_RADIUS,
                      in_fan=C.IN_FAN, exc_frac=C.EXC_FRAC,
                      refractory=C.REFRACTORY, seed=C.SEED, **kw)
            m_tr, j_tr = run_pool(lsm, spk_tr)
            m_te, j_te = run_pool(lsm, spk_te)
            pd = eval_detection(m_tr, j_tr, m_te, j_te,
                                spk_tr.mean(-1), spk_te.mean(-1), y_tr, y_te)
            rmse = eval_doppler(m_tr, j_tr, m_te, j_te, y_tr, y_te, fd_tr, fd_te)
            print(f"{tag:>22s} | Pd={pd:.3f} | RMSE={rmse:.4f} | rate={lsm.last_spike_rate:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
