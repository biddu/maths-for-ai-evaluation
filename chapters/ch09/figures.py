"""Chapter 9 figures (greyscale-safe, 6x9 trim)."""
import json, pathlib, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.special import betaln, comb

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]
num = json.loads((HERE / "numbers.json").read_text())
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

d = pd.read_csv(ROOT / "data" / "humaneval_k20.csv")
truth = pd.read_csv(ROOT / "data" / "humaneval_k20_truth.csv")
counts = d.groupby(["model", "item"])["pass"].sum().unstack(0)
n = num["n_samples"]


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def beta_pass_at_k(a, b, k):
    return 1.0 - math.exp(betaln(a, b + k) - betaln(a, b))


# Figure 9.1: heterogeneity of the per-item pass rates, with the fitted Beta
fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.4), sharey=True)
for ax, mdl in zip(axes, ("A", "B")):
    ph = counts[mdl].to_numpy() / n
    ax.hist(ph, bins=np.linspace(0, 1, 21), color="0.75", edgecolor="white", density=True)
    a, b = num[mdl]["beta_ml"]
    x = np.linspace(0.005, 0.995, 300)
    ax.plot(x, beta.pdf(x, a, b), color="black", lw=1.2)
    ax.set_title(f"model {mdl}: Beta({a:.2f}, {b:.2f})", fontsize=8.5)
    ax.set_xlabel(r"$\hat p_i$ from 20 samples")
    ax.set_xlim(0, 1)
axes[0].set_ylabel("density")
fig.tight_layout(w_pad=1.5); fig.savefig(HERE / "fig_heterogeneity.pdf")

# Figure 9.2: pass@k and pass^k (left); the failure rate 1 - pass@k and the extrapolations (right), model A
fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.7))
ks = np.arange(1, 21)
c = counts["A"].to_numpy(); ph = c / n
est = np.array([np.mean([pass_at_k(n, ci, k) for ci in c]) for k in ks])
pk = np.array([np.mean([comb(ci, k) / comb(n, k) if ci >= k else 0.0 for ci in c]) for k in ks])
a, b = num["A"]["beta_ml"]
kk = np.arange(1, 401)
beta_curve = np.array([beta_pass_at_k(a, b, k) for k in kk])
plug_curve = np.array([np.mean(1 - (1 - ph) ** k) for k in kk])
pt = truth["p_A"].to_numpy()
true_curve = np.array([np.mean(1 - (1 - pt) ** k) for k in kk])
ax = axes[0]
ax.plot(ks, est, "o", color="black", ms=3, label="pass@$k$")
ax.plot(ks, pk, "s", color="black", ms=2.8, mfc="white", label="pass$^k$")
ax.plot(kk[:20], beta_curve[:20], color="black", lw=0.9)
ax.plot(kk[:20], [math.exp(betaln(a + k, b) - betaln(a, b)) for k in kk[:20]], color="black", lw=0.9, ls=":")
ax.set_xlim(0, 21); ax.set_ylim(0, 1.02); ax.set_xlabel("$k$"); ax.set_ylabel("probability")
ax.legend(frameon=False, fontsize=7.5, loc="center right")
ax = axes[1]
ax.plot(ks, 1 - est, "o", color="black", ms=3, label="unbiased, $n = 20$")
ax.plot(kk, 1 - beta_curve, color="black", lw=1.1, label="Beta extrapolation")
ax.plot(kk, 1 - plug_curve, color="black", lw=1.0, ls="--", label="plug-in")
ax.plot(kk, 1 - true_curve, color="0.6", lw=2.2, zorder=0, label="truth")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.9, 400); ax.set_ylim(1e-4, 1)
ax.set_xlabel("$k$ (log scale)"); ax.set_ylabel("$1 - $pass@$k$ (log scale)")
ax.legend(frameon=False, fontsize=7, loc="upper right")
fig.tight_layout(w_pad=1.2); fig.savefig(HERE / "fig_passk_curves.pdf")
print("figures written")
