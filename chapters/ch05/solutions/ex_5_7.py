import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, warnings
from worked_example import balanced_components, se_cluster, cluster_bootstrap, ROOT
he = pd.read_csv(ROOT / "data" / "humaneval_k20.csv")
XB = he[he.model == "B"].pivot(index="item", columns="sample", values="pass").to_numpy()
for label, X in (("20 samples", XB), ("first 5", XB[:, :5])):
    c = balanced_components(X)
    print(f"{label}: sb2={c['sb2']:.3f} sw2={c['sw2']:.3f} icc={c['icc']:.2f} deff={c['deff']:.2f} "
          f"SE={100*se_cluster(X):.2f} boot={100*cluster_bootstrap(X, seed=2).std(ddof=1):.2f} points")
import statsmodels.formula.api as smf
long = he[he.model == "B"][["item", "pass"]].rename(columns={"pass": "y"})
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fit = smf.mixedlm("y ~ 1", long, groups=long["item"]).fit(reml=True)
print(f"MixedLM SE = {100*fit.bse['Intercept']:.2f} points")
