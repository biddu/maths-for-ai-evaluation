import math, numpy as np
from scipy.stats import t as tdist
rng = np.random.default_rng(11); Z = 1.959964; icc, p, k = 0.5, 0.7, 5
tau2 = icc * p * (1 - p); a = p * (p * (1 - p) / tau2 - 1); b = a * (1 - p) / p
for C in (5, 10, 30, 100):
    cn = cz = ct = 0; reps = 4000
    for _ in range(reps):
        pc = rng.beta(a, b, C); X = (rng.random((C, k)) < pc[:, None]).astype(float); ph = X.mean()
        se_n = math.sqrt(ph * (1 - ph) / X.size) if 0 < ph < 1 else 0.0
        se_c = X.mean(axis=1).std(ddof=1) / math.sqrt(C)
        cn += abs(ph - p) <= Z * se_n; cz += abs(ph - p) <= Z * se_c
        ct += abs(ph - p) <= tdist.ppf(0.975, C - 1) * se_c
    print(f"C={C:3d}  naive {cn/reps:.3f}  cluster-z {cz/reps:.3f}  cluster-t {ct/reps:.3f}")
