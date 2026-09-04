import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from worked_example import brier_decomposition, ROOT
g = pd.read_csv(ROOT / "data" / "mmlu_probs.csv")
conf, corr = g["confidence"].to_numpy(), g["correct"].to_numpy(float)
for b in (5, 10, 20, 50):
    print(b, "equal-width ECE", round(brier_decomposition(conf, corr, b)["ece"], 4))
order = np.argsort(conf); ece = sum(len(c) * abs(conf[c].mean() - corr[c].mean()) for c in np.array_split(order, 10))
print("10 equal-count bins ECE", round(ece / conf.size, 4))
lg = np.log(conf / (1 - conf)); tr, te = slice(0, 1000), slice(1000, 2000)
best = None
for a in np.linspace(0.3, 1.2, 181):
    for b in np.linspace(-1, 0.5, 151):
        q = 1 / (1 + np.exp(-(a * lg[tr] + b)))
        ll = -np.mean(corr[tr] * np.log(q) + (1 - corr[tr]) * np.log(1 - q))
        if best is None or ll < best[0]:
            best = (ll, a, b)
_, a, b = best
q = 1 / (1 + np.exp(-(a * lg[te] + b)))
print(f"held-out: a={a:.2f} b={b:.2f} Brier={((q - corr[te])**2).mean():.4f} ECE10={brier_decomposition(q, corr[te], 10)['ece']:.4f}")
