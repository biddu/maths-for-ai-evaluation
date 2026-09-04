"""Chapter 8 figures (greyscale-safe, 6x9 trim)."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
num = json.loads((HERE / "numbers.json").read_text())
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})


def floor_ratio(theta, s, t):
    """SE of the corrected estimate at N -> infinity divided by the human-only SE,
    same calibration size; independent of n_h."""
    J = s + t - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt((theta * s * (1 - s) + (1 - theta) * t * (1 - t)) / (J ** 2 * theta * (1 - theta)))


# Figure 8.1: regime map in the (sensitivity, specificity) plane at theta = 0.5
s = np.linspace(0.5, 1, 301); t = np.linspace(0.5, 1, 301)
S, T = np.meshgrid(s, t)
R = floor_ratio(0.5, S, T)
fig, ax = plt.subplots(figsize=(3.6, 3.3))
ax.contourf(S, T, R, levels=[0, 1], colors=["0.85"])
cs = ax.contour(S, T, R, levels=[0.4, 0.6, 0.8, 1.0, 1.5, 2.0], colors="black", linewidths=[0.6, 0.6, 0.6, 1.3, 0.6, 0.6])
ax.clabel(cs, fmt="%.1f", fontsize=7, inline=True)
for name, mk in (("J", "s"), ("L", "o")):
    j = num[f"judge_{name}"]
    ax.plot(j["s_true"], j["t_true"], marker=mk, color="black", ms=6, mfc="white")
    ax.annotate(name, (j["s_true"], j["t_true"]), xytext=(-9, -11), textcoords="offset points", fontsize=8.5)
ax.set_xlabel("sensitivity $s$"); ax.set_ylabel("specificity $t$")
ax.set_xlim(0.5, 1); ax.set_ylim(0.5, 1); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(HERE / "fig_regime.pdf")

# Figure 8.2: standard error against calibration-set size, N = 1800 judged items
N = num["N_judged_only"]
nh = np.arange(40, 1201)
theta = 0.5
fig, ax = plt.subplots(figsize=(4.4, 2.8))
ax.plot(nh, np.sqrt(theta * (1 - theta) / nh) * 100, color="black", lw=1.3, label="human labels only")
for name, ls in (("J", "--"), ("L", ":")):
    j = num[f"judge_{name}"]; s_, t_ = j["s_true"], j["t_true"]; J = s_ + t_ - 1
    q = s_ * theta + (1 - t_) * (1 - theta)
    var = (q * (1 - q) / N + (theta * s_ * (1 - s_) + (1 - theta) * t_ * (1 - t_)) / nh) / J ** 2
    ax.plot(nh, np.sqrt(var) * 100, color="black", lw=1.1, ls=ls, label=f"corrected, judge {name}")
ax.axvline(200, color="0.6", lw=0.8)
ax.set_xlabel("calibration items $n_h$ (with $N = 1\\,800$ judged items)")
ax.set_ylabel("standard error (points)")
ax.set_ylim(0, 9); ax.set_xlim(0, 1200)
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(HERE / "fig_se_vs_nh.pdf")
print("figures written")
