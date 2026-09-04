"""Exercise 7.8: exact match and kappa for judge J across prevalences by subsampling."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from worked_example import agreement_table, cohen_kappa, kappa_se, kappa_max, ROOT

j = pd.read_csv(ROOT / "data" / "judge_labels.csv")
T = agreement_table(j["human"], j["J"], [0, 1])
k, po, pe = cohen_kappa(T)
print(f"exact match {po:.3f}, kappa {k:.3f} (SE {kappa_se(T):.3f}), kappa_max {kappa_max(T)[0]:.3f}")
rng = np.random.default_rng(78)
neg = j.index[j.human == 0].to_numpy(); pos = j.index[j.human == 1].to_numpy()
for prev in (0.5, 0.6, 0.7, 0.8, 0.85):
    n_pos = int(round(neg.size * prev / (1 - prev)))
    ms, ks = [], []
    for _ in range(200):
        sub = np.concatenate([neg, rng.choice(pos, n_pos, replace=False)])
        Ts = agreement_table(j.loc[sub, "human"], j.loc[sub, "J"], [0, 1])
        kk, pp, _ = cohen_kappa(Ts); ms.append(pp); ks.append(kk)
    print(f"prevalence {prev:.2f} (n = {neg.size + n_pos}): exact match {np.mean(ms):.3f}, kappa {np.mean(ks):.3f}")
