"""Chapter 11 figures (greyscale-safe, 6x9 trim)."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
num = json.loads((HERE / "numbers.json").read_text())
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
MODELS = num["models"]; K = len(MODELS)

# Figure 11.1: Bradley-Terry strengths with intervals (left) and Elo in three orders (right)
fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.7), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
beta = np.array(num["mm"]["beta"]); se = np.array(num["mm"]["se"])
lo = np.array(num["boot"]["lo"]); hi = np.array(num["boot"]["hi"])
order = np.argsort(-beta)
for y, k in enumerate(order):
    ax.errorbar(beta[k], y, xerr=1.96 * se[k], fmt="o", color="black", ms=3.5, elinewidth=1.0, capsize=2.5)
    ax.plot(num["true_beta"][k], y, marker="|", color="black", ms=9, mew=1.2)
ax.set_yticks(range(K)); ax.set_yticklabels([MODELS[k] for k in order]); ax.invert_yaxis()
ax.set_xlabel(r"Bradley--Terry strength $\hat\beta$")
ax.set_xlim(-1.6, 1.1); ax.set_ylim(6.5, -0.5)
ax.text(-1.55, 6.25, "95% intervals; tick marks the truth", fontsize=6.5, color="0.3")
ax = axes[1]
orders = ["as_played", "reversed", "shuffled"]
labels = ["played", "reversed", "shuffled"]
for k, m in enumerate(MODELS):
    r = [num["elo"][o]["ratings"][m] for o in orders]
    ax.plot(range(3), r, "-o", color="black", lw=0.8, ms=3, mfc="white" if k in (1, 2, 3, 4) else "black")
    ax.text(2.1, r[-1], m, fontsize=7, va="center")
for k, m in enumerate(MODELS):
    ax.plot(-0.35, num["elo"]["bt_as_elo"][m], marker="_", color="0.4", ms=8, mew=1.5)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=7)
ax.set_xlim(-0.6, 2.6); ax.set_ylabel("Elo after 300 battles ($K = 32$)")
ax.text(-0.5, 1150, "BT\n(Elo scale)", fontsize=6.3, color="0.4", ha="left")
fig.tight_layout(w_pad=1.0); fig.savefig(HERE / "fig_bt_elo.pdf")

# Figure 11.2: standard error of the largest contested gap against battles, uniform vs greedy
fig, ax = plt.subplots(figsize=(3.6, 2.4))
cp = num["active"]["checkpoints"]
ax.plot(cp, num["active"]["uniform_mean_se"], color="black", lw=1.2, label="uniform random pairs")
ax.plot(cp, num["active"]["greedy_mean_se"], color="black", lw=1.2, ls="--", label="greedy (largest gain)")
ax.set_xlabel("battles"); ax.set_ylabel("SE of the largest gap, M2--M5")
ax.set_ylim(0, 0.8); ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(HERE / "fig_active.pdf")
print("figures written")
