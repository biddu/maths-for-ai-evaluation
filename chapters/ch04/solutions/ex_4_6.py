import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from worked_example import bootstrap, bca_ci, percentile_ci, ROOT
m = pd.read_csv(ROOT / "data" / "mtbench_judge.csv")
f = m[m.model == "F"].sort_values("question")["score"].to_numpy(float)
for name, st in (("mean", np.mean), ("median", np.median), ("frac10", lambda x: (x == 10).mean())):
    b = bootstrap(st, f, seed=5)
    lo, hi, z0, a, _, _ = bca_ci(st, f, b)
    print(f"{name:7s} percentile {percentile_ci(b)}  BCa ({lo:.4f}, {hi:.4f})  z0={z0:.3f} a={a:.3f}")
