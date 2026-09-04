import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from worked_example import power_mcnemar, paired_bootstrap, ROOT
rng = np.random.default_rng(0)
n, delta = 1319, 0.02
for psi in (0.10, 0.20, 0.30):
    p10, p01 = (psi + delta) / 2, (psi - delta) / 2
    draws = rng.multinomial(n, [1 - psi, p10, p01], 20_000)
    b, c = draws[:, 1], draws[:, 2]
    stat = (b - c) ** 2 / np.maximum(b + c, 1)
    print(psi, (stat > 3.841).mean(), power_mcnemar(n, delta, psi))
g = pd.read_csv(ROOT / "data" / "gsm8k_pairs.csv")
d = paired_bootstrap(g["C"], g["D"], seed=2)
print(np.quantile(d, [0.025, 0.975]), d.std(ddof=1))
