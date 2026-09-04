"""Exercise 9.7: Beta fits for model B, extrapolation against the truth, and refitting on 5 samples."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from worked_example import beta_fit_counts, beta_fit_moments, beta_pass_at_k, ROOT
d = pd.read_csv(ROOT / "data" / "humaneval_k20.csv"); truth = pd.read_csv(ROOT / "data" / "humaneval_k20_truth.csv")
pt = truth["p_B"].to_numpy()
wide = d[d.model == "B"].pivot(index="item", columns="sample", values="pass").to_numpy()
for n in (20, 5):
    c = wide[:, :n].sum(axis=1); ph = c / n
    a1, b1 = beta_fit_counts(c, n); a2, b2 = beta_fit_moments(c, n)
    print(f"n = {n}: ML Beta({a1:.2f}, {b1:.2f}), moments Beta({a2:.2f}, {b2:.2f}), items with a pass {np.mean(c > 0):.3f}")
    for k in (50, 100, 200):
        print(f"   k={k:3d}: ML {beta_pass_at_k(a1, b1, k):.4f}  moments {beta_pass_at_k(a2, b2, k):.4f}  "
              f"plug-in {np.mean(1 - (1 - ph) ** k):.4f}  truth {np.mean(1 - (1 - pt) ** k):.4f}")
