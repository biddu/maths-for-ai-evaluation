"""Chapter 5 figures (greyscale-safe, 6x9 trim)."""
import json, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]
num = json.loads((HERE / "numbers.json").read_text())
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

# Figure 5.1: per-item pass rates for model A over 20 samples
he = pd.read_csv(ROOT / "data" / "humaneval_k20.csv")
XA = he[he.model == "A"].pivot(index="item", columns="sample", values="pass").to_numpy()
m = XA.mean(axis=1)
fig, ax = plt.subplots(figsize=(4.6, 2.4))
ax.hist(m, bins=np.linspace(0, 1, 21), color="0.75", edgecolor="white")
ax.axvline(m.mean(), color="black", lw=1.0, ls=":")
ax.text(m.mean() - 0.02, ax.get_ylim()[1] * 0.9, f"mean {m.mean():.2f}", ha="right", fontsize=7.5)
ax.set_xlabel("per-item pass rate over 20 samples, model A")
ax.set_ylabel("number of items")
fig.tight_layout(); fig.savefig(HERE / "fig_item_rates.pdf")

# Figure 5.2: SE of the score against samples per item, correct vs naive
ks = np.array([1, 2, 3, 5, 10, 20, 50, 100, 1000])
n = num["A_n"]; sb2, sw2 = num["A_sb2"], num["A_sw2"]; p = num["A_p_all"]
kk = np.logspace(0, 3, 200)
correct = 100 * np.sqrt((sb2 + sw2 / kk) / n)
naive = 100 * np.sqrt(p * (1 - p) / (n * kk))
fig, ax = plt.subplots(figsize=(4.6, 2.7))
ax.plot(kk, correct, color="black", lw=1.2, label="correct: $\\sqrt{(\\sigma_b^2 + \\sigma_w^2/k)/n}$")
ax.plot(kk, naive, color="black", lw=1.2, ls="--", label="naive: $\\sqrt{p(1-p)/nk}$")
ax.axhline(100 * np.sqrt(sb2 / n), color="0.6", lw=0.8, ls=":")
ax.text(12, 100 * np.sqrt(sb2 / n) - 0.28, f"floor {100*np.sqrt(sb2/n):.2f} points (items only)", fontsize=7.5, color="0.35")
ax.set_xscale("log"); ax.set_xlabel("samples per item $k$ (164 items)"); ax.set_ylabel("standard error (points)")
ax.set_ylim(0, 4); ax.legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig(HERE / "fig_se_vs_k.pdf")
print("figures written")
