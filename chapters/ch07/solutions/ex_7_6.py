"""Exercise 7.6: kappa family against sklearn/statsmodels; the concordance identity; bootstrap."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import itertools
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import cohens_kappa
from worked_example import agreement_table, cohen_kappa, kappa_se, weight_matrix, concordance_cc, bootstrap_kappa, ROOT

d = pd.read_csv(ROOT / "data" / "raters300.csv")
q = d.pivot(index="item", columns="rater", values="quality").dropna().astype(int)
cats = [1, 2, 3, 4, 5]
for a, b in itertools.combinations(["R1", "R2", "R3"], 2):
    T = agreement_table(q[a], q[b], cats)
    for kind, skw in ((None, None), ("linear", "linear"), ("quadratic", "quadratic")):
        k = cohen_kappa(T, kind)[0]
        assert abs(k - cohen_kappa_score(q[a], q[b], weights=skw)) < 1e-12
        res = cohens_kappa(T, weights=1 - weight_matrix(5, kind), return_results=True)
        assert abs(math.sqrt(res.var_kappa) - kappa_se(T, kind)) < 1e-9
    kq = cohen_kappa(T, "quadratic")[0]
    assert abs(kq - concordance_cc(q[a], q[b])) < 1e-12
    print(f"{a}-{b}: all checks pass; quadratic kappa_w = CCC = {kq:.6f}")
T = agreement_table(q["R1"], q["R3"], cats)
boot = bootstrap_kappa(q["R1"].to_numpy(), q["R3"].to_numpy(), cats, "quadratic", reps=4000, seed=13)
print(f"R1-R3 quadratic: formula SE {kappa_se(T, 'quadratic'):.4f}, bootstrap SE {boot.std(ddof=1):.4f}")
