# -*- coding: utf-8 -*-
"""Regenerate beam-scan figure including the new 2D CNN ceiling."""
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"D:\kimi_workspace\lidar-pointnet\snn\outputs_isal"
TFLN_RESULTS = r"D:\kimi_workspace\tfln-dispersion-lab\results\snn"

d = np.load(os.path.join(TFLN_RESULTS, "m4_beam_scan_results.npz"),
            allow_pickle=True)
keys = ["single_random_ESN", "single_random_CNN", "4random_avg_CNN",
        "4beam_avg_CNN", "4beam_concat_CE", "4beam_seq_ESN",
        "4beam_seq_ESN+angle", "4beam_3step_ESN",
        "4beam_3step_shuffled_ESN", "4beam_3step_2DCNN_ceiling",
        "2beam_seq_ESN", "8beam_seq_ESN"]
vals = [float(d[k]) for k in keys]

colors = (["C0"] * 2 + ["C1"] + ["C2"] * 9)
fig, ax = plt.subplots(figsize=(10, 5.2))
bars = ax.bar(range(len(keys)), vals, color=colors)
ax.set_xticks(range(len(keys)), keys, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("test accuracy")
ax.set_ylim(0, 1)
ax.set_title("Multi-beam sector-limited ordered scanning + empirical ceiling (D=4 m)")
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, "%.2f" % v, ha="center", fontsize=7)
ax.axhline(0.1, color="gray", ls="--", lw=0.8, alpha=0.5)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "m4_fig_beam_scan.png"), dpi=130)
plt.close(fig)
for f in ["m4_fig_beam_scan.png"]:
    shutil.copy(os.path.join(OUT, f), os.path.join(TFLN_RESULTS, f))
print("figure updated with ceiling %.3f" % float(d["4beam_3step_2DCNN_ceiling"]))
