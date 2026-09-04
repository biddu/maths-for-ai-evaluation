"""Chapter 6 figures (greyscale-safe, 6x9 trim)."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

HERE = pathlib.Path(__file__).parent
num = json.loads((HERE / "numbers.json").read_text())
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

# Figure 6.1: the leaderboard with intervals and rank intervals
order = num["observed_order"]
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for y, m in enumerate(order):
    s, se = num["scores"][m], num["ses_points"][m]
    ri = num["rank_intervals"][m]
    ax.errorbar(s, y, xerr=1.96 * se, fmt="o", color="black", ms=3.5, elinewidth=1.0, capsize=2.5)
    ax.text(79.5, y, f"rank {ri['observed_rank']:2d}: [{ri['lo']}, {ri['hi']}]", fontsize=7.5, va="center", family="monospace")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.invert_yaxis()
ax.set_xlim(46, 86); ax.set_xticks(range(50, 81, 5)); ax.set_xlabel("macro-average score (%) with 95% interval")
ax.axvspan(79, 86, color="white", lw=0)
for sp in ("left",): ax.spines[sp].set_visible(True)
fig.tight_layout(); fig.savefig(HERE / "fig_leaderboard.pdf")

# Figure 6.2: expected maximum of m equal models, in standard-error units
ms = np.arange(1, 101)
emax = [quad(lambda x, m=m: x * m * norm.pdf(x) * norm.cdf(x) ** (m - 1), -12, 12)[0] for m in ms]
fig, ax = plt.subplots(figsize=(4.6, 2.6))
ax.plot(ms, emax, color="black", lw=1.2, label="independent scores (theory)")
ax.plot([12], [num["null_max_excess_in_se"]], marker="s", color="black", ms=5, ls="none",
        label="12 models on shared items (permutation)")
ax.set_xscale("log"); ax.set_xlabel("number of models compared, $m$")
ax.set_ylabel("expected excess of the top score\n(standard errors)")
ax.set_ylim(0, 2.8); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(HERE / "fig_winners_curse.pdf")
print("figures written")
