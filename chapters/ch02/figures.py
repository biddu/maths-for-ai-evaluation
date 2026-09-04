"""Chapter 2 figures (greyscale-safe, 6x9 trim)."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
num = json.loads((HERE / "numbers.json").read_text())
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

# Figure 2.1: exact coverage at n = 50 for three intervals
grid = np.array(num["coverage_grid"])
curves = num["coverage_curves_n50"]
fig, axes = plt.subplots(3, 1, figsize=(4.6, 5.2), sharex=True)
for ax, (key, label) in zip(axes, (("wald", "Wald"), ("wilson", "Wilson"), ("clopper_pearson", "Clopper--Pearson"))):
    ax.plot(grid, curves[key], color="black", lw=0.9)
    ax.axhline(0.95, color="0.55", lw=0.8, ls="--")
    ax.set_ylim(0.6, 1.005)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "0.95", "1"])
    ax.text(0.02, 0.63, label.replace("--", "–"), fontsize=9)
axes[-1].set_xlabel("population rate $p$")
axes[1].set_ylabel("coverage probability of the nominal 95% interval")
fig.tight_layout(h_pad=0.6)
fig.savefig(HERE / "fig_coverage_n50.pdf")

# Figure 2.2: the five intervals for 49 of 50
names = [("wald", "Wald"), ("agresti_coull", "Agresti–Coull"), ("wilson", "Wilson"),
         ("jeffreys", "Jeffreys"), ("clopper_pearson", "Clopper–Pearson")]
fig, ax = plt.subplots(figsize=(4.6, 2.5))
for y, (key, label) in enumerate(names):
    lo, hi = num[f"49of50_{key}_lo"], num[f"49of50_{key}_hi"]
    ax.plot([100 * lo, 100 * hi], [y, y], color="black", lw=2.2, solid_capstyle="butt")
    ax.text(100 * lo - 0.4, y, f"{100*lo:.1f}", ha="right", va="center", fontsize=7.5)
    ax.text(100 * hi + 0.4, y, f"{100*hi:.1f}", ha="left", va="center", fontsize=7.5)
ax.axvline(98, color="0.45", lw=0.8, ls=":")
ax.axvline(100, color="0.45", lw=0.8)
ax.axvspan(100, 104, color="0.92", lw=0)
ax.set_yticks(range(len(names))); ax.set_yticklabels([l for _, l in names])
ax.set_xlim(84, 104); ax.set_xticks([84, 88, 92, 96, 100])
ax.set_xlabel("95% interval for the pass rate (%), 49 passes out of 50")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(HERE / "fig_five_intervals.pdf")
print("figures written")
