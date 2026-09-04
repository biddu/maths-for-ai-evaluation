"""Chapter 3 figures (greyscale-safe, 6x9 trim)."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
num = json.loads((HERE / "numbers.json").read_text())
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

# Figure 3.1: sampling distribution of the difference, paired vs independent
paired = 100 * np.load(HERE / "sim_paired.npy")
indep = 100 * np.load(HERE / "sim_indep.npy")
bins = np.arange(-20, 21, 1.22) - 0.61
fig, ax = plt.subplots(figsize=(4.6, 2.7))
ax.hist(indep, bins=bins, density=True, color="0.82", edgecolor="0.82", label="independent items (SD 5.0)")
ax.hist(paired, bins=bins, density=True, histtype="step", color="black", lw=1.2, label="same items, paired (SD 4.2)")
ax.axvline(num["he_diff_points"], color="black", ls=":", lw=0.9)
ax.text(6.5, 0.083, "observed\ndifference 1.2", fontsize=7.5, va="top")
ax.set_xlabel("difference in HumanEval score, model A minus model B (points)")
ax.set_ylabel("density")
ax.set_xlim(-20, 20)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(HERE / "fig_paired_vs_indep.pdf")

# Figure 3.2: items needed for 80% power against the discordance rate
grid = np.array(num["psi_grid"])
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for delta, ls in ((0.02, "-"), (0.03, "--"), (0.05, ":")):
    y = np.array([v if v is not None else np.nan for v in num[f"n_curve_delta{delta}"]], float)
    ax.plot(grid, y, color="black", ls=ls, lw=1.2, label=f"difference {100*delta:.0f} points")
for name, size in (("HumanEval", 164), ("GSM8K", 1319), ("MMLU", 14042)):
    ax.axhline(size, color="0.6", lw=0.7)
    ax.text(0.49, size * 1.12, name, fontsize=7.5, color="0.35", ha="right")
ax.set_yscale("log")
ax.set_xlabel("discordance rate $\\psi$ (fraction of items the two models disagree on)")
ax.set_ylabel("items for 80% power")
ax.set_xlim(0, 0.5)
ax.set_ylim(80, 30000)
ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(0.0, 0.66))
fig.tight_layout()
fig.savefig(HERE / "fig_n_vs_psi.pdf")
print("figures written")

# Figure 3.3: the score statistic Z(delta) for the GSM8K table, with Wald and Tango limits
grid = np.array(num["score_grid_points"])
curve = np.array(num["score_curve_gsm"])
fig, ax = plt.subplots(figsize=(4.6, 2.7))
ax.plot(grid, curve, color="black", lw=1.2)
ax.axhline(1.96, color="0.55", lw=0.8, ls="--"); ax.axhline(-1.96, color="0.55", lw=0.8, ls="--")
ax.axhline(0, color="0.8", lw=0.6)
for x in (num["gsm_tango_lo_points"], num["gsm_tango_hi_points"]):
    ax.axvline(x, color="black", lw=0.8, ls=":")
for x in (num["gsm_ci_wald_lo_points"], num["gsm_ci_wald_hi_points"]):
    ax.plot([x], [0], marker="|", color="black", ms=9, mew=1.2)
ax.text(num["gsm_tango_lo_points"] + 0.15, -3.5, "Tango lower", fontsize=7.5, ha="left")
ax.text(num["gsm_tango_hi_points"] + 0.15, 3.3, "Tango upper", fontsize=7.5, ha="left")
ax.text(5.2, 0.25, "Wald limits (ticks)", fontsize=7.5)
ax.text(-1.9, 2.15, "$+1.96$", fontsize=7.5); ax.text(-1.9, -2.45, "$-1.96$", fontsize=7.5)
ax.set_xlabel("candidate difference $\\delta$ (points), model C minus model D")
ax.set_ylabel("score statistic $Z(\\delta)$")
ax.set_xlim(-2, 7); ax.set_ylim(-4.5, 4.5)
fig.tight_layout()
fig.savefig(HERE / "fig_score_curve.pdf")
print("figure 3.3 written")
