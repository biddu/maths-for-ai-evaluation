"""Exercise 8.7: standard errors against calibration size, four estimators, judge L."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from worked_example import rogan_gladen, rg_variance, ppi_mean, ppi_lambda, observed_rate, ROOT
d = pd.read_csv(ROOT / "data" / "judge2000.csv")
f_all = d.L.to_numpy(); y_all = d.truth.to_numpy()      # truth used as the human label here
rng = np.random.default_rng(87)
M = len(d)
s_true, t_true, theta = 0.96, 0.90, 0.5
J = s_true + t_true - 1; q = observed_rate(theta, s_true, t_true)
A = q * (1 - q) / J ** 2; B = (theta * s_true * (1 - s_true) + (1 - theta) * t_true * (1 - t_true)) / J ** 2
print("n_h   human    RG     PPI    PPI++")
for n_h in (50, 100, 200, 400, 800):
    out = np.zeros(4)
    for _ in range(200):
        idx = rng.permutation(M); c, r = idx[:n_h], idx[n_h:]
        y, fc, f = y_all[c], f_all[c], f_all[r]
        n1, n0 = int(y.sum()), int(n_h - y.sum())
        s, t = fc[y == 1].mean(), 1 - fc[y == 0].mean()
        se_h = math.sqrt(y.mean() * (1 - y.mean()) / n_h)
        se_rg = math.sqrt(rg_variance(f.mean(), f.size, s, n1, t, n0))
        se_pp = ppi_mean(f, fc, y)[1]
        se_pl = ppi_mean(f, fc, y, ppi_lambda(f, fc, y))[1]
        out += np.array([se_h, se_rg, se_pp, se_pl]) / 200
    print(f"{n_h:4d}  " + "  ".join(f"{v:.4f}" for v in out))
# analytic crossing with N = 2000 - n_h:  n_h < (2000 - n_h) (0.25 - B) / A
k = (0.25 - B) / A
print(f"analytic crossing n_h = {2000 * k / (1 + k):.0f}")
