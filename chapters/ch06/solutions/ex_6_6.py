import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, numpy as np
from statsmodels.stats.multitest import multipletests
from worked_example import bonferroni, holm, benjamini_hochberg
num = json.loads((pathlib.Path(__file__).resolve().parents[1] / "numbers.json").read_text())
p = np.array([v["p"] for v in num["pairwise"].values()])
for name, f, meth in (("bonferroni", bonferroni, "bonferroni"), ("holm", holm, "holm"), ("bh", benjamini_hochberg, "fdr_bh")):
    assert (f(p) == multipletests(p, 0.05, meth)[0]).all()
    print(name, int(f(p).sum()))
rng = np.random.default_rng(0)
u = rng.random(10_000)
for f, meth in ((bonferroni, "bonferroni"), (holm, "holm"), (benjamini_hochberg, "fdr_bh")):
    assert (f(u) == multipletests(u, 0.05, meth)[0]).all()
fw = np.zeros(4)
for _ in range(20_000):
    q = rng.random(66)
    fw += [(q <= 0.05).any(), bonferroni(q).any(), holm(q).any(), benjamini_hochberg(q).any()]
print("FWER under 66 true nulls: raw, bonferroni, holm, bh =", np.round(fw / 20_000, 3))
