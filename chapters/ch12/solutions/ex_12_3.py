"""Exercise 12.3: recompute the capstone family with a chosen baseline (default M1)."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import norm
from worked_example import (two_prop_z, mcnemar_exact, holm, cluster_se, ppi_mean, pass_at_k, ROOT, MODELS)

def family(base):
    know = pd.read_csv(ROOT / "data" / "capstone_knowledge.csv"); read = pd.read_csv(ROOT / "data" / "capstone_reading.csv")
    judged = pd.read_csv(ROOT / "data" / "capstone_judged.csv"); code = pd.read_csv(ROOT / "data" / "capstone_code.csv")
    others = [m for m in MODELS if m != base]; tests = []
    n_k = len(know)
    for m in others:
        _, pn = two_prop_z(know[m].sum(), n_k, know[base].sum(), n_k)
        b = int(((know[m] == 1) & (know[base] == 0)).sum()); c = int(((know[m] == 0) & (know[base] == 1)).sum())
        tests.append(("know", m, pn, mcnemar_exact(b, c), (know[m] - know[base]).mean()))
    n_r = len(read)
    for m in others:
        _, pn = two_prop_z(read[m].sum(), n_r, read[base].sum(), n_r)
        d = (read[m] - read[base]).to_numpy(float); se, _ = cluster_se(d, read["passage"].to_numpy())
        tests.append(("read", m, pn, 2 * norm.sf(abs(d.mean() / se)), d.mean()))
    est = {}
    for m in MODELS:
        g = judged[judged.model == m]; gl = g.dropna(subset=["human"]); gu = g[g["human"].isna()]
        est[m] = ppi_mean(gu["judge"].to_numpy(), gl["judge"].to_numpy(), gl["human"].astype(int).to_numpy())[:2]
    for m in others:
        gm, gb = judged[judged.model == m], judged[judged.model == base]
        _, pn = two_prop_z(gm["judge"].sum(), len(gm), gb["judge"].sum(), len(gb))
        d = est[m][0] - est[base][0]; se = math.sqrt(est[m][1] ** 2 + est[base][1] ** 2)
        tests.append(("judge", m, pn, 2 * norm.sf(abs(d / se)), d))
    wide = {m: code.pivot(index="item", columns="sample", values=m).to_numpy() for m in MODELS}
    v5 = {m: np.array([pass_at_k(10, c, 5) for c in wide[m].sum(1)]) for m in MODELS}
    n_c = len(v5[base])
    for m in others:
        a5, b5 = wide[m][:, :5].max(1), wide[base][:, :5].max(1)
        _, pn = two_prop_z(a5.sum(), n_c, b5.sum(), n_c)
        d = v5[m] - v5[base]; se = d.std(ddof=1) / math.sqrt(n_c)
        tests.append(("code", m, pn, 2 * norm.sf(abs(d.mean() / se)), d.mean()))
    ph = np.array([t[3] for t in tests]); pn = np.array([t[2] for t in tests]); H = holm(ph)
    print(f"baseline {base}: naive stars {(pn < 0.05).sum()}, honest uncorrected {(ph < 0.05).sum()}, Holm survivors {H.sum()}")
    for t, h in zip(tests, H):
        print(f"   {t[0]:6s} {t[1]} diff {t[4]:+.3f} naive p {t[2]:.3f}{'*' if t[2] < 0.05 else ' '} honest p {t[3]:.4f} {'H' if h else ''}")

family("M3"); family("M1")
