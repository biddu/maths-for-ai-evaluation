"""Chapter 4 figures (greyscale-safe, 6x9 trim)."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
num = json.loads((HERE / "numbers.json").read_text())
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

# Figure 4.1: distribution of judge grades for the two models
fig, ax = plt.subplots(figsize=(4.6, 2.5))
x = np.arange(1, 11)
ax.bar(x - 0.2, num["E_hist"], width=0.4, color="black", label=f"model E (mean {num['E_mean']:.2f})")
ax.bar(x + 0.2, num["F_hist"], width=0.4, color="0.7", label=f"model F (mean {num['F_mean']:.2f})")
ax.set_xticks(x); ax.set_xlabel("judge grade (1--10)".replace("--", "–")); ax.set_ylabel("number of questions")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(HERE / "fig_grades.pdf")

# Figure 4.2: reliability diagrams, raw and recalibrated, 10 bins
fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.6), sharey=True)
for ax, key, title in ((axes[0], "mmlu_bins10", "as reported"), (axes[1], "recal_bins10", "after recalibration")):
    rows = num[key]
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
    confs = [r["conf"] for r in rows]; accs = [r["acc"] for r in rows]; ns = [r["n"] for r in rows]
    ax.plot(confs, accs, color="black", marker="o", ms=3.5, lw=1.0)
    for c_, a_, n_ in zip(confs, accs, ns):
        if n_ >= 20:
            ax.text(c_, a_ - 0.09, str(n_), fontsize=6, ha="center", color="0.35")
    ax.set_xlim(0.2, 1.0); ax.set_ylim(0.2, 1.0)
    ax.set_xlabel("stated confidence (bin mean)"); ax.set_title(title, fontsize=9)
axes[0].set_ylabel("observed accuracy in bin")
fig.tight_layout(); fig.savefig(HERE / "fig_reliability.pdf")

# Figure 4.3: ECE against number of bins
fig, ax = plt.subplots(figsize=(4.6, 2.5))
g = num["ece_bin_grid"]
ax.plot(g, num["ece_curve_raw"], color="black", lw=1.2, label="as reported")
ax.plot(g, num["ece_curve_recal"], color="black", lw=1.2, ls="--", label="after recalibration")
ax.set_xlabel("number of equal-width bins"); ax.set_ylabel("ECE")
ax.set_ylim(0, 0.16); ax.set_xlim(0, 60)
ax.legend(frameon=False, fontsize=7.5, loc="center right")
fig.tight_layout(); fig.savefig(HERE / "fig_ece_bins.pdf")
print("figures written")
