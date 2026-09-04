"""Exercise 11.6: MM against the logistic form, the position term, and the bootstrap."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from worked_example import win_matrix, bt_mm, design, logistic_fit, bootstrap_bt, MODELS, ROOT
df = pd.read_csv(ROOT / "data" / "arena300.csv")
beta, _, _ = bt_mm(win_matrix(df)); X, y = design(df)
b0, _, _, _ = logistic_fit(X, y, drop_position=True); b1, gamma, cov, _ = logistic_fit(X, y)
print("max |MM - logistic|:", np.max(np.abs(beta - b0)))
print(f"gamma {gamma:.3f} (SE {np.sqrt(cov[-1, -1]):.3f}); max strength change {np.max(np.abs(b1 - beta)):.3f}")
print("order without / with position:", [MODELS[k] for k in np.argsort(-beta)], [MODELS[k] for k in np.argsort(-b1)])
rng = np.random.default_rng(116); betas, ranks = bootstrap_bt(df, 2000, rng)
for k, m in enumerate(MODELS):
    print(f"{m}: beta {beta[k]:6.3f}  boot SE {betas[:, k].std(ddof=1):.3f}  rank [{int(np.quantile(ranks[:, k], .025))}, {int(np.quantile(ranks[:, k], .975))}]  P(first) {(ranks[:, k] == 1).mean():.3f}")
