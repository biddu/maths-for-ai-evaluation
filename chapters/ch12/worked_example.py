"""Chapter 12 -- every number printed in the chapter, regenerated from data.

Run:  python chapters/ch12/worked_example.py
Writes chapters/ch12/numbers.json.

The capstone rebuilds a leaderboard-style table for models M1..M5 (baseline M3)
from the five capstone_*.csv files, first the naive way and then with the
methods of Chapters 1-11.
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm, binom

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).with_name("numbers.json")
Z = norm.ppf(0.975)
MODELS = [f"M{i}" for i in range(1, 6)]
BASE = "M3"
OTHERS = [m for m in MODELS if m != BASE]
S_L, T_L = 0.96, 0.90          # judge L's operating point on this task, from the calibration set


# ---------------------------------------------------------------------------
# Tools from earlier chapters, restated compactly
# ---------------------------------------------------------------------------
def wilson(x, n, z=Z):
    p = x / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def two_prop_z(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2; p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se, 2 * norm.sf(abs(p1 - p2) / se)


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * binom.cdf(min(b, c), n, 0.5))


def holm(p, alpha=0.05):
    p = np.asarray(p); m = p.size; order = np.argsort(p); reject = np.zeros(m, bool)
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


def cluster_se(values, clusters):
    """SE of the grand mean from cluster means (balanced clusters)."""
    df = pd.DataFrame({"v": values, "c": clusters}); means = df.groupby("c")["v"].mean()
    return float(means.std(ddof=1) / math.sqrt(means.size)), means


def ppi_mean(f_unlab, f_lab, y_lab):
    N, n = len(f_unlab), len(f_lab)
    f_unlab, f_lab, y_lab = (np.asarray(a, float) for a in (f_unlab, f_lab, y_lab))
    cov = np.cov(f_lab, y_lab, ddof=1)[0, 1]; vf = np.var(np.concatenate([f_unlab, f_lab]), ddof=1)
    lam = cov / vf * N / (N + n) if vf > 0 else 0.0
    rect = y_lab - lam * f_lab
    est = lam * f_unlab.mean() + rect.mean()
    var = lam ** 2 * f_unlab.var(ddof=1) / N + rect.var(ddof=1) / n
    return est, math.sqrt(var), lam


def rogan_gladen(q, s, t):
    return (q + t - 1) / (s + t - 1)


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def bt_mm(W, iters=2000, tol=1e-10):
    K = W.shape[0]; N = W + W.T; w = W.sum(1); pi = np.ones(K)
    for _ in range(iters):
        denom = np.array([sum(N[i, j] / (pi[i] + pi[j]) for j in range(K) if j != i) for i in range(K)])
        new = w / denom; new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(np.log(new) - np.log(pi))) < tol:
            pi = new; break
        pi = new
    return np.log(pi) - np.log(pi).mean()


def bt_fisher_cov(beta, W):
    K = W.shape[0]; N = W + W.T; I = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            if N[i, j] > 0:
                p = 1 / (1 + math.exp(beta[j] - beta[i])); v = N[i, j] * p * (1 - p)
                I[i, i] += v; I[j, j] += v; I[i, j] -= v; I[j, i] -= v
    return np.linalg.pinv(I)


def win_matrix(df):
    idx = {m: k for k, m in enumerate(MODELS)}; W = np.zeros((5, 5))
    for _, r in df.iterrows():
        i, j = idx[r["first"]], idx[r["second"]]
        if r["outcome"] == "first": W[i, j] += 1
        elif r["outcome"] == "second": W[j, i] += 1
        else: W[i, j] += 0.5; W[j, i] += 0.5
    return W


def elo(df, k_factor=32):
    R = {m: 1000.0 for m in MODELS}
    for _, r in df.iterrows():
        a, b = r["first"], r["second"]; s = {"first": 1.0, "second": 0.0, "tie": 0.5}[r["outcome"]]
        e = 1 / (1 + 10 ** ((R[b] - R[a]) / 400)); R[a] += k_factor * (s - e); R[b] -= k_factor * (s - e)
    return R


def main():
    num = {"z": Z, "models": MODELS, "baseline": BASE}
    rng = np.random.default_rng(12)
    know = pd.read_csv(ROOT / "data" / "capstone_knowledge.csv")
    read = pd.read_csv(ROOT / "data" / "capstone_reading.csv")
    judged = pd.read_csv(ROOT / "data" / "capstone_judged.csv")
    code = pd.read_csv(ROOT / "data" / "capstone_code.csv")
    battles = pd.read_csv(ROOT / "data" / "capstone_battles.csv")
    naive, honest = {m: {} for m in MODELS}, {m: {} for m in MODELS}
    tests = []      # (metric, model, naive_p, honest_p, naive_diff, honest_diff, honest_se)

    # --- knowledge: 500 shared items ---------------------------------------------------------
    n_k = len(know)
    for m in MODELS:
        x = int(know[m].sum()); naive[m]["know"] = x / n_k
        lo, hi = wilson(x, n_k); honest[m]["know"] = {"p": x / n_k, "lo": lo, "hi": hi, "n": n_k}
    for m in OTHERS:
        _, p_naive = two_prop_z(know[m].sum(), n_k, know[BASE].sum(), n_k)
        b = int(((know[m] == 1) & (know[BASE] == 0)).sum()); c = int(((know[m] == 0) & (know[BASE] == 1)).sum())
        d = (know[m] - know[BASE]).to_numpy(float)
        tests.append(("know", m, p_naive, mcnemar_exact(b, c), d.mean(), d.mean(), d.std(ddof=1) / math.sqrt(n_k)))
        honest[m]["know"].update({"b": b, "c": c})

    # --- reading: 40 passages x 8 questions --------------------------------------------------
    n_r = len(read); n_pass = read["passage"].nunique()
    for m in MODELS:
        x = int(read[m].sum()); naive[m]["read"] = x / n_r
        se_c, means = cluster_se(read[m].to_numpy(), read["passage"].to_numpy())
        se_n = math.sqrt((x / n_r) * (1 - x / n_r) / n_r)
        honest[m]["read"] = {"p": x / n_r, "se": se_c, "lo": x / n_r - Z * se_c, "hi": x / n_r + Z * se_c,
                             "se_naive": se_n, "deff": (se_c / se_n) ** 2, "n": n_r, "clusters": n_pass}
    for m in OTHERS:
        _, p_naive = two_prop_z(read[m].sum(), n_r, read[BASE].sum(), n_r)
        d = (read[m] - read[BASE]).to_numpy(float)
        se_c, _ = cluster_se(d, read["passage"].to_numpy())
        z = d.mean() / se_c
        tests.append(("read", m, p_naive, 2 * norm.sf(abs(z)), d.mean(), d.mean(), se_c))

    # --- judged: 400 responses per model, judge L, 100 human labels each ---------------------
    # calibration of judge L pooled over the five models' human-labelled subsets
    lab = judged.dropna(subset=["human"]); lab_y = lab["human"].astype(int)
    s_hat = float(lab.loc[lab_y == 1, "judge"].mean()); t_hat = float(1 - lab.loc[lab_y == 0, "judge"].mean())
    num["judge_calib"] = {"n": int(len(lab)), "n1": int((lab_y == 1).sum()), "n0": int((lab_y == 0).sum()),
                          "s_hat": s_hat, "t_hat": t_hat, "youden": s_hat + t_hat - 1,
                          "s_se": math.sqrt(s_hat * (1 - s_hat) / (lab_y == 1).sum()),
                          "t_se": math.sqrt(t_hat * (1 - t_hat) / (lab_y == 0).sum())}
    for m in MODELS:
        g = judged[judged.model == m]
        q = float(g["judge"].mean()); naive[m]["judge"] = q
        gl = g.dropna(subset=["human"]); gu = g[g["human"].isna()]
        est, se, lam = ppi_mean(gu["judge"].to_numpy(), gl["judge"].to_numpy(), gl["human"].astype(int).to_numpy())
        honest[m]["judge"] = {"q": q, "q_se": math.sqrt(q * (1 - q) / len(g)), "ppi": est, "se": se, "lam": lam,
                              "lo": est - Z * se, "hi": est + Z * se, "rg": rogan_gladen(q, s_hat, t_hat),
                              "human_only": float(gl["human"].astype(int).mean()), "truth": float(g["truth"].mean()),
                              "N": int(len(gu)), "n_h": int(len(gl))}
    for m in OTHERS:
        gm, gb = judged[judged.model == m], judged[judged.model == BASE]
        _, p_naive = two_prop_z(gm["judge"].sum(), len(gm), gb["judge"].sum(), len(gb))
        d = honest[m]["judge"]["ppi"] - honest[BASE]["judge"]["ppi"]
        se = math.sqrt(honest[m]["judge"]["se"] ** 2 + honest[BASE]["judge"]["se"] ** 2)
        tests.append(("judge", m, p_naive, 2 * norm.sf(abs(d / se)), naive[m]["judge"] - naive[BASE]["judge"], d, se))

    # --- code: 100 items x 10 samples; naive pass@5 from the first 5 samples ----------------------
    n_c = code["item"].nunique()
    wide = {m: code.pivot(index="item", columns="sample", values=m).to_numpy() for m in MODELS}
    per_item = {}
    for m in MODELS:
        any5 = wide[m][:, :5].max(1); naive[m]["pass5"] = float(any5.mean())
        cnt = wide[m].sum(1)
        v5 = np.array([pass_at_k(10, c, 5) for c in cnt]); v1 = cnt / 10
        per_item[m] = v5
        boot = np.array([v5[rng.integers(0, n_c, n_c)].mean() for _ in range(2000)])
        honest[m]["code"] = {"pass1": float(v1.mean()), "pass1_se": float(v1.std(ddof=1) / math.sqrt(n_c)),
                             "pass5": float(v5.mean()), "pass5_se": float(v5.std(ddof=1) / math.sqrt(n_c)),
                             "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975)),
                             "naive_any5": float(any5.mean()), "n_items": n_c, "n_samples": 10}
    for m in OTHERS:
        a5, b5 = wide[m][:, :5].max(1), wide[BASE][:, :5].max(1)
        _, p_naive = two_prop_z(a5.sum(), n_c, b5.sum(), n_c)
        d = per_item[m] - per_item[BASE]; se = d.std(ddof=1) / math.sqrt(n_c)
        tests.append(("code", m, p_naive, 2 * norm.sf(abs(d.mean() / se)), float(a5.mean() - b5.mean()), float(d.mean()), float(se)))

    # --- preference: 400 battles ------------------------------------------------------------------
    W = win_matrix(battles); beta = bt_mm(W); cov = bt_fisher_cov(beta, W)
    R = elo(battles)
    for k, m in enumerate(MODELS):
        naive[m]["elo"] = R[m]
        honest[m]["bt"] = {"beta": float(beta[k]), "se": float(math.sqrt(cov[k, k]))}
    # bootstrap ranks
    B = 2000; ranks = np.empty((B, 5), int); nb = len(battles)
    for r in range(B):
        bb = bt_mm(win_matrix(battles.iloc[rng.integers(0, nb, nb)]), iters=300, tol=1e-8)
        ranks[r] = (-bb).argsort().argsort() + 1
    for k, m in enumerate(MODELS):
        honest[m]["bt"].update({"rank": int((-beta).argsort().argsort()[k] + 1),
                                "rank_lo": int(np.quantile(ranks[:, k], 0.025)), "rank_hi": int(np.quantile(ranks[:, k], 0.975)),
                                "p_first": float((ranks[:, k] == 1).mean())})
    kb = MODELS.index(BASE)
    for m in OTHERS:
        k = MODELS.index(m)
        gap = float(beta[k] - beta[kb]); se = math.sqrt(cov[k, k] + cov[kb, kb] - 2 * cov[k, kb])
        # naive: the Elo ranking makes a claim whenever the ratings differ; treat any Elo gap as a "claim"
        tests.append(("bt", m, 0.0 if R[m] != R[BASE] else 1.0, 2 * norm.sf(abs(gap / se)), (R[m] - R[BASE]), gap, se))
    num["elo_order"] = sorted(MODELS, key=lambda m: -R[m])
    num["bt_order"] = [MODELS[k] for k in (-beta).argsort()]
    num["plausibly_best"] = [m for m in MODELS if honest[m]["bt"]["p_first"] > 0.025]

    # --- the claims: naive stars vs Holm-corrected survivors -------------------------------------
    accuracy_tests = [t for t in tests if t[0] != "bt"]
    p_naive = np.array([t[2] for t in accuracy_tests]); p_honest = np.array([t[3] for t in accuracy_tests])
    naive_sig = p_naive < 0.05
    honest_raw = p_honest < 0.05
    honest_holm = holm(p_honest)
    num["claims"] = {"family_size": int(len(accuracy_tests)),
                     "naive_significant": int(naive_sig.sum()), "honest_uncorrected": int(honest_raw.sum()),
                     "honest_holm": int(honest_holm.sum()),
                     "naive_that_survive_holm": int((naive_sig & honest_holm).sum()),
                     "naive_that_fail": int((naive_sig & ~honest_holm).sum()),
                     "new_under_honest": int((~naive_sig & honest_holm).sum())}
    num["tests"] = [{"metric": t[0], "model": t[1], "p_naive": t[2], "p_honest": t[3], "diff_naive": t[4],
                     "diff_honest": t[5], "se_honest": t[6],
                     "naive_star": bool(t[2] < 0.05), "holm": bool(h)}
                    for t, h in zip(accuracy_tests, honest_holm)]
    bt_tests = [t for t in tests if t[0] == "bt"]
    num["bt_tests"] = [{"model": t[1], "gap": t[5], "se": t[6], "p": t[3]} for t in bt_tests]
    num["naive"] = naive; num["honest"] = honest
    # summaries used in the text
    num["read_deff_range"] = [min(honest[m]["read"]["deff"] for m in MODELS), max(honest[m]["read"]["deff"] for m in MODELS)]
    num["judge_gap_naive_vs_ppi"] = {m: {"naive": naive[m]["judge"], "ppi": honest[m]["judge"]["ppi"], "truth": honest[m]["judge"]["truth"]} for m in MODELS}

    OUT.write_text(json.dumps(num, indent=2))
    print(json.dumps(num["claims"]))
    for t in num["tests"]:
        print(f"{t['metric']:6s} {t['model']}  naive p {t['p_naive']:.4f} {'*' if t['naive_star'] else ' '}   honest p {t['p_honest']:.4f} {'H' if t['holm'] else ' '}   diff {t['diff_honest']:+.3f} ({t['se_honest']:.3f})")
    for t in num["bt_tests"]:
        print("bt", t)
    print("elo order", num["elo_order"], "bt order", num["bt_order"], "plausibly best", num["plausibly_best"])
    for m in MODELS:
        h = honest[m]
        print(m, f"know {h['know']['p']:.3f} [{h['know']['lo']:.3f},{h['know']['hi']:.3f}]  read {h['read']['p']:.3f} se {h['read']['se']:.3f} deff {h['read']['deff']:.2f}  "
                 f"judge naive {h['judge']['q']:.3f} ppi {h['judge']['ppi']:.3f} ({h['judge']['se']:.3f}) rg {h['judge']['rg']:.3f} truth {h['judge']['truth']:.3f}  "
                 f"pass5 naive {h['code']['naive_any5']:.3f} unbiased {h['code']['pass5']:.3f} ({h['code']['pass5_se']:.3f})  bt {h['bt']['beta']:+.2f} rank {h['bt']['rank']} [{h['bt']['rank_lo']},{h['bt']['rank_hi']}] elo {naive[m]['elo']:.0f}")
    print("judge calib", num["judge_calib"])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
