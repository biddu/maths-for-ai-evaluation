"""Chapter 12 figure (greyscale-safe, 6x9 trim)."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
num = json.loads((HERE / "numbers.json").read_text())
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

# Figure 12.1: every comparison with the baseline, naive and honest, with 95% intervals
metrics = [("know", "knowledge"), ("read", "reading"), ("judge", "judge L (PPI)"), ("code", "pass@5")]
models = ["M1", "M2", "M4", "M5"]
fig, axes = plt.subplots(1, 4, figsize=(4.5, 2.6), sharey=True)
for ax, (key, title) in zip(axes, metrics):
    rows = [t for t in num["tests"] if t["metric"] == key]
    for y, m in enumerate(models):
        t = next(r for r in rows if r["model"] == m)
        d, se = 100 * t["diff_honest"], 100 * t["se_honest"]
        ax.errorbar(d, y, xerr=1.96 * se, fmt="o", color="black", ms=3.2, elinewidth=1.0, capsize=2.5,
                    mfc="black" if t["holm"] else "white")
        ax.plot(100 * t["diff_naive"], y + 0.32, marker="v", color="0.55", ms=3.5, ls="none")
        if t["naive_star"]:
            ax.text(100 * t["diff_naive"], y + 0.34, "*", fontsize=9, ha="center", va="bottom", color="0.3")
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_title(title, fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(models)
    ax.set_xlim(-22, 22); ax.set_xticks([-15, 0, 15])
    ax.tick_params(labelsize=7.5)
axes[0].set_ylabel("model, against baseline M3"); axes[0].invert_yaxis()
fig.text(0.5, 0.01, "difference from M3 (points). Filled: survives Holm. Grey triangle: naive difference; * naive star.",
         ha="center", fontsize=6.6, color="0.3")
fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=0.6); fig.savefig(HERE / "fig_claims.pdf")
print("figure written")
