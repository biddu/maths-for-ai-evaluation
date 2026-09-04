"""Chapter 10 figures (greyscale-safe, 6x9 trim)."""
import json, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]
num = json.loads((HERE / "numbers.json").read_text())
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
P0, PER_DAY, CHANGE = num["p0"], num["per_day"], num["change_day"]
zc = norm.ppf(1 - num["alpha"])

# Figure 10.1: the peeking problem. Left: P(at least one rejection by day T) under the null.
# Right: three null paths of the cumulative z statistic with the rejection line.
rng = np.random.default_rng(1010)
fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.4))
ax = axes[0]
Ts = np.arange(1, 366)
counts = rng.binomial(PER_DAY, P0, size=(4000, 365))
S = counts.cumsum(1); N = PER_DAY * np.arange(1, 366)
z = (S - N * P0) / np.sqrt(N * P0 * (1 - P0))
ever = np.minimum.accumulate(z, axis=1) < -zc
ax.plot(Ts, ever.mean(0), color="black", lw=1.2)
ax.axhline(0.05, color="0.6", lw=0.8, ls="--")
ax.set_xscale("log"); ax.set_xlim(1, 365); ax.set_ylim(0, 0.45)
ax.set_xlabel("days of daily testing $T$ (log scale)"); ax.set_ylabel("P(rejected at least once)")
ax.text(1.15, 0.065, r"$\alpha = 0.05$", fontsize=7.5, color="0.4")
ax = axes[1]
crossers = np.where(ever[:, 99])[0][:2]          # two null paths that reject within 100 days
paths = [crossers[0], crossers[1], np.where(~ever[:, 99])[0][0]]
for r, ls in zip(paths, ("-", "--", ":")):
    ax.plot(np.arange(1, 101), z[r, :100], color="black", lw=0.9, ls=ls)
ax.axhline(-zc, color="0.5", lw=0.9)
ax.set_xlabel("day"); ax.set_ylabel("cumulative $z$")
ax.set_ylim(-3.2, 3.2); ax.set_xlim(0, 100)
ax.text(78, -zc + 0.12, "reject below", fontsize=7, color="0.4")
fig.tight_layout(w_pad=1.2); fig.savefig(HERE / "fig_peeking.pdf")

# Figure 10.2: the daily stream with the three monitors
d = pd.read_csv(ROOT / "data" / "daily_stream.csv")
daily = d.groupby("day")["pass"].sum().to_numpy() / PER_DAY
days = np.arange(1, 101)
mon = num["monitor"]["data"]
fig, axes = plt.subplots(3, 1, figsize=(4.5, 4.6), sharex=True)
ax = axes[0]
ax.plot(days, daily, "o", color="black", ms=2)
ax.fill_between(days, mon["cs_lo"], mon["cs_hi"], color="0.85", lw=0)
ax.plot(days, np.cumsum(daily) / days, color="black", lw=1.0)
ax.axhline(P0, color="0.5", lw=0.9, ls="--")
ax.set_ylabel("pass rate"); ax.set_ylim(0.65, 0.9)
ax.text(99, 0.875, "daily rate (dots), running mean (line),\n95% confidence sequence (band)", fontsize=7, color="0.3", ha="right", va="top")
ax = axes[1]
ax.plot(days, mon["z_path"], color="black", lw=1.0)
ax.axhline(-zc, color="0.5", lw=0.9)
zmin = min(mon["z_path"])
ax.set_ylabel("cumulative $z$"); ax.set_ylim(zmin - 1, 2.5)
ax.text(2, zmin - 0.3, "daily one-sided test at 0.05", fontsize=7, color="0.3")
ax = axes[2]
ax.plot(days, num["cusum"]["path"], color="black", lw=1.0)
ax.axhline(num["cusum"]["h_star"], color="0.5", lw=0.9)
smax = max(num["cusum"]["path"])
ax.set_ylabel("CUSUM $S_t$"); ax.set_xlabel("day"); ax.set_ylim(0, smax * 1.1)
ax.text(2, smax, f"threshold $h = {num['cusum']['h_star']:.1f}$", fontsize=7, color="0.3")
for ax in axes:
    ax.axvline(CHANGE, color="0.3", lw=0.7, ls=":")
fig.tight_layout(h_pad=0.6); fig.savefig(HERE / "fig_stream.pdf")
print("figures written")
