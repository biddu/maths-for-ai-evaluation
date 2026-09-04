"""Exercise 7.5: SE of kappa at three prevalences for raters with the skills of R1 and R2."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import norm
from worked_example import cohen_kappa, kappa_se

Z = norm.ppf(0.975)
s1, s2, t1, t2 = 0.80, 0.75, 0.97, 0.96
for prev in (0.02, 0.08, 0.30):
    a = prev * s1 * s2 + (1 - prev) * (1 - t1) * (1 - t2)
    d = prev * (1 - s1) * (1 - s2) + (1 - prev) * t1 * t2
    b = prev * s1 * (1 - s2) + (1 - prev) * (1 - t1) * t2
    c = prev * (1 - s1) * s2 + (1 - prev) * t1 * (1 - t2)
    T = 300 * np.array([[d, c], [b, a]])          # rows: rater 1 (0,1); cols: rater 2 (0,1)
    k, po, pe = cohen_kappa(T)
    se = kappa_se(T)
    n_needed = 300 * (Z * se / 0.10) ** 2
    print(f"prev {prev:.2f}: kappa {k:.3f}  SE(n=300) {se:.3f}  n for half-width 0.10: {n_needed:,.0f}  expected positives {300*prev:.0f}")
