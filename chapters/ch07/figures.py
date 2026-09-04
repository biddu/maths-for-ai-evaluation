"""Chapter 7 figures (greyscale-safe, 6x9 trim)."""
import json, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]
num = json.loads((HERE / "numbers.json").read_text())
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})


def kappa_prev(prev, sa, sb, ta, tb):
    a = prev * sa * sb + (1 - prev) * (1 - ta) * (1 - tb)
    d = prev * (1 - sa) * (1 - sb) + (1 - prev) * ta * tb
    pa = prev * sa + (1 - prev) * (1 - ta)
    pb = prev * sb + (1 - prev) * (1 - tb)
    po = a + d
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe), po


# Figure 7.1: raw agreement and kappa against prevalence, two fixed pairs of raters
prev = np.linspace(0.01, 0.5, 200)
fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.5), sharey=True)
for ax, (sa, sb, ta, tb, title) in zip(axes, [
        (0.75, 0.70, 0.96, 0.95, "two human raters (R2, R3)"),
        (num["judge"]["sens"], num["judge"]["spec"], 1.0, 1.0, None)]):
    if title is None:
        # the judge against the human reference: reference is exact (sens = spec = 1)
        s, t = num["judge"]["sens"], num["judge"]["spec"]
        ks, pos = [], []
        for u in prev:
            p = 1 - u                                # p = share acceptable
            a = p * s; d = (1 - p) * t
            po = a + d; pj = p * s + (1 - p) * (1 - t)
            pe = pj * p + (1 - pj) * (1 - p)
            ks.append((po - pe) / (1 - pe)); pos.append(po)
        ks, pos = np.array(ks), np.array(pos)
        title = "judge J against the human label"
        xprev = prev                # judge data: the rare class is 'unacceptable'
        ax.plot(xprev, pos, color="black", lw=1.2, label="exact match")
        ax.plot(xprev, ks, color="black", lw=1.2, ls="--", label=r"Cohen's $\kappa$")
        ax.axvline(1 - num["judge"]["prevalence"], color="0.6", lw=0.8)
        ax.set_xlabel("share unacceptable")
    else:
        ks, pos = zip(*[kappa_prev(p, sa, sb, ta, tb) for p in prev])
        ax.plot(prev, pos, color="black", lw=1.2, label="raw agreement $p_o$")
        ax.plot(prev, ks, color="black", lw=1.2, ls="--", label=r"Cohen's $\kappa$")
        ax.axvline(0.08, color="0.6", lw=0.8)
        ax.set_xlabel("prevalence of the flag")
    ax.set_ylim(0, 1); ax.set_xlim(0, 0.5)
    ax.set_title(title, fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
axes[0].set_ylabel("agreement")
fig.tight_layout(w_pad=1.5); fig.savefig(HERE / "fig_kappa_prevalence.pdf")

# Figure 7.2: the 5x5 agreement table for quality, R1 against R3, as a bubble plot,
# with the quadratic weights shaded behind it
d = pd.read_csv(ROOT / "data" / "raters300.csv")
q = d.pivot(index="item", columns="rater", values="quality").dropna()
T = np.zeros((5, 5))
for a, b in zip(q["R1"].astype(int), q["R3"].astype(int)):
    T[a - 1, b - 1] += 1
i, j = np.indices((5, 5))
W = 1 - ((i - j) / 4) ** 2
fig, ax = plt.subplots(figsize=(3.4, 3.2))
ax.imshow(1 - W, cmap="Greys", vmin=-0.15, vmax=2.0, origin="lower")
for a in range(5):
    for b in range(5):
        if T[a, b] > 0:
            ax.scatter(b, a, s=5.5 * T[a, b], color="black", zorder=3)
            ax.text(b, a - 0.36, f"{int(T[a, b])}", ha="center", va="center", fontsize=6.5, color="black", zorder=4)
ax.set_xticks(range(5)); ax.set_xticklabels(range(1, 6)); ax.set_yticks(range(5)); ax.set_yticklabels(range(1, 6))
ax.set_xlabel("R3 quality"); ax.set_ylabel("R1 quality")
for sp in ("top", "right"): ax.spines[sp].set_visible(True)
ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 4.5)
fig.tight_layout(); fig.savefig(HERE / "fig_quality_table.pdf")
print("figures written")
