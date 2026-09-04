"""Chapter 1 figures (greyscale-safe, 6x9 trim)."""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

# Figure 1.1: standard error against n for three values of p
n = np.logspace(1, 4.3, 300)
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for p, ls in ((0.5, "-"), (0.8, "--"), (0.95, ":")):
    ax.plot(n, 100 * np.sqrt(p * (1 - p) / n), color="black", ls=ls, lw=1.2, label=f"$p={p}$")
for name, size in (("HumanEval", 164), ("GSM8K", 1319), ("MMLU", 14042)):
    ax.axvline(size, ymin=0, ymax=0.62, color="0.6", lw=0.7)
    ax.text(size * 1.08, 6.7, name, fontsize=7.5, color="0.35", rotation=90, va="top")
ax.set_xscale("log")
ax.set_xlabel("number of test items $n$")
ax.set_ylabel("standard error (points)")
ax.set_ylim(0, 11)
ax.legend(frameon=False, loc="upper right", ncol=3)
fig.tight_layout()
fig.savefig(HERE / "fig_se_vs_n.pdf")

# Figure 1.2: the two reported scores with +/- 1 SE and +/- 2 SE bars
import json
num = json.loads((HERE / "numbers.json").read_text())
fig, ax = plt.subplots(figsize=(4.6, 2.2))
for y, (lab, p, se) in enumerate((("Model A", num["p_A"], num["se_A"]),
                                   ("Model B", num["p_B"], num["se_B"]))):
    ax.errorbar(100 * p, y, xerr=100 * 2 * se, fmt="none", ecolor="0.7", elinewidth=4, capsize=0)
    ax.errorbar(100 * p, y, xerr=100 * se, fmt="o", color="black", ecolor="black", elinewidth=1.4, capsize=3, ms=4)
    ax.text(100 * p, y + 0.28, f"{100*p:.1f}", ha="center", fontsize=8)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Model A", "Model B"])
ax.set_ylim(-0.6, 1.7); ax.set_xlim(60, 82); ax.set_xticks(range(60, 83, 4))
ax.set_xlabel("HumanEval score (%), with $\\pm 1$ SE (black) and $\\pm 2$ SE (grey)")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(HERE / "fig_two_scores.pdf")
print("figures written")
