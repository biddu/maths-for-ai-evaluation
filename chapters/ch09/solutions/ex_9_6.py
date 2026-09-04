"""Exercise 9.6: the product-form estimator against Monte Carlo, and Table 9.1."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from worked_example import pass_at_k, pass_pow_k, item_bootstrap, ROOT
rng = np.random.default_rng(96)
for n, c, k in ((20, 6, 5), (20, 2, 10), (200, 3, 50)):
    # draw p from a Beta near the target count, then simulate
    p = rng.beta(c + 1, n - c + 1, 20000)
    cs = rng.binomial(n, p)
    est = np.array([pass_at_k(n, ci, k) for ci in cs]); truth = 1 - (1 - p) ** k
    print(f"(n, c, k) = ({n}, {c}, {k}): estimator mean {est.mean():.4f}, E[1-(1-p)^k] {truth.mean():.4f}, "
          f"estimate at c = {c}: {pass_at_k(n, c, k):.4f}")
d = pd.read_csv(ROOT / "data" / "humaneval_k20.csv")
counts = d.groupby(["model", "item"])["pass"].sum().unstack(0)
print("\n k     A pass@k (SE)   [boot]          B pass@k (SE)   [boot]")
for k in range(1, 21):
    row = []
    for mdl in ("A", "B"):
        v = np.array([pass_at_k(20, ci, k) for ci in counts[mdl]])
        se = v.std(ddof=1) / math.sqrt(v.size); bse, lo, hi = item_bootstrap(v, reps=2000, seed=k)
        row.append(f"{v.mean():.3f} ({se:.3f}) [{lo:.3f}, {hi:.3f}]")
    print(f"{k:2d}   " + "   ".join(row))
