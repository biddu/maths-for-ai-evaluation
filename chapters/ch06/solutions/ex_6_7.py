import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from worked_example import ROOT, MODELS
df = pd.read_csv(ROOT / "data" / "leaderboard12.csv")
X = df[MODELS].to_numpy(); bench = df["benchmark"].to_numpy()
bidx = {b: np.where(bench == b)[0] for b in sorted(set(bench))}
rng = np.random.default_rng(1); B = 2000
def ranks(independent):
    out = np.zeros((B, 12), int)
    for r in range(B):
        sc = np.zeros(12)
        for idx in bidx.values():
            if independent:
                for j in range(12):
                    sc[j] += X[idx[rng.integers(0, idx.size, idx.size)], j].mean() / 6
            else:
                samp = idx[rng.integers(0, idx.size, idx.size)]
                sc += X[samp].mean(axis=0) / 6
        out[r] = (-sc).argsort().argsort() + 1
    return out
for label, ind in (("together", False), ("independent", True)):
    rk = ranks(ind)
    print(label, [f"{m}:[{int(np.quantile(rk[:,j],0.025))},{int(np.quantile(rk[:,j],0.975))}]" for j, m in enumerate(MODELS)])
# winner's curse with independent noise around model M6's outcomes
base = X[:, 5].astype(float); se = base.std() / np.sqrt(base.size)
for m in (5, 12, 100):
    ex = []
    for _ in range(2000):
        scores = [base[rng.integers(0, base.size, base.size)].mean() for _ in range(m)]
        ex.append((max(scores) - base.mean()) / se)
    print(m, round(float(np.mean(ex)), 2))
