"""Exercise 8.6: Rogan-Gladen with delta-method and bootstrap intervals on judge2000."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import norm
from worked_example import rogan_gladen, rg_variance, ROOT
Z = norm.ppf(0.975)
d = pd.read_csv(ROOT / "data" / "judge2000.csv")
cal, rest = d[d.calib == 1], d[d.calib == 0]
y = cal.human.astype(int).to_numpy()
rng = np.random.default_rng(86)
for jn in ("J", "L"):
    fc = cal[jn].to_numpy()
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    s, t = fc[y == 1].mean(), 1 - fc[y == 0].mean()
    for label, judged in (("disjoint", rest), ("all 2000", d)):
        N = len(judged); q = judged[jn].mean()
        th = rogan_gladen(q, s, t); se = math.sqrt(rg_variance(q, N, s, n1, t, n0))
        boots = np.array([rogan_gladen(rng.binomial(N, q) / N, rng.binomial(n1, s) / n1, rng.binomial(n0, t) / n0) for _ in range(4000)])
        print(f"judge {jn} ({label}): theta {th:.4f} SE {se:.4f} delta [{th - Z * se:.3f}, {th + Z * se:.3f}] "
              f"boot [{np.quantile(boots, 0.025):.3f}, {np.quantile(boots, 0.975):.3f}]  truth {judged.truth.mean():.4f}")
