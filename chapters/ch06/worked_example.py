"""Chapter 6 -- every number printed in the chapter, regenerated from data.

Run:  python chapters/ch06/worked_example.py
Writes chapters/ch06/numbers.json.
"""
import itertools
import json
import math
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
from statsmodels.stats.multitest import multipletests

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).with_name("numbers.json")
Z = norm.ppf(0.975)
MODELS = [f"M{j}" for j in range(1, 13)]


# ---------------------------------------------------------------------------
# From-scratch procedures (Sections 6.2-6.3)
# ---------------------------------------------------------------------------
def bonferroni(p, alpha=0.05):
    p = np.asarray(p)
    return p <= alpha / p.size


def holm(p, alpha=0.05):
    p = np.asarray(p)
    m = p.size
    order = np.argsort(p)
    reject = np.zeros(m, bool)
    for rank, idx in enumerate(order):          # rank 0 is the smallest p
        if p[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break                                # stop at the first failure
    return reject


def benjamini_hochberg(p, alpha=0.05):
    p = np.asarray(p)
    m = p.size
    order = np.argsort(p)
    sorted_p = p[order]
    thresholds = alpha * np.arange(1, m + 1) / m
    below = np.where(sorted_p <= thresholds)[0]
    reject = np.zeros(m, bool)
    if below.size:
        k = below.max()                          # largest i with p_(i) <= alpha i/m
        reject[order[: k + 1]] = True
    return reject


def expected_max_normal(m):
    """E[max of m iid standard normals] by numerical integration."""
    f = lambda x: x * m * norm.pdf(x) * norm.cdf(x) ** (m - 1)
    return quad(f, -12, 12)[0]


def leaderboard(df):
    """Macro-average score per model (equal weight per benchmark) and its SE."""
    scores, ses = {}, {}
    for mdl in MODELS:
        pb = df.groupby("benchmark")[mdl].mean()
        nb = df.groupby("benchmark")[mdl].size()
        scores[mdl] = float(pb.mean())
        ses[mdl] = float(math.sqrt(((pb * (1 - pb) / nb) ** 1).sum()) / len(pb))
    return scores, ses


def paired_diff(df, a, b):
    """Macro-average difference a - b and its paired SE across benchmarks."""
    diff, var = 0.0, 0.0
    groups = df.groupby("benchmark")
    B = groups.ngroups
    for _, g in groups:
        w = (g[a] - g[b]).to_numpy(float)
        diff += w.mean() / B
        var += w.var(ddof=1) / w.size / B ** 2
    return diff, math.sqrt(var)


def main():
    num = {"z": Z}
    df = pd.read_csv(ROOT / "data" / "leaderboard12.csv")
    sizes = df.groupby("benchmark").size()
    num["benchmarks"] = int(sizes.size)
    num["sizes"] = {k: int(v) for k, v in sizes.items()}
    num["total_items"] = int(sizes.sum())

    # --- 6.1 the multiplicity arithmetic ------------------------------------
    for m in (1, 6, 15, 28, 66, 96, 396):
        num[f"fwer_indep_m{m}"] = 1 - 0.95 ** m
        num[f"bonf_bound_m{m}"] = min(1.0, 0.05 * m)
    num["pairs_12"] = 66
    num["bonf_alpha_66"] = 0.05 / 66
    num["by_factor_66"] = float(sum(1 / i for i in range(1, 67)))

    # --- 6.5 leaderboard scores and pairwise tests --------------------------------
    scores, ses = leaderboard(df)
    order = sorted(MODELS, key=lambda k: -scores[k])
    num["scores"] = {k: 100 * scores[k] for k in MODELS}
    num["ses_points"] = {k: 100 * ses[k] for k in MODELS}
    num["observed_order"] = order
    num["top"] = order[0]
    pairs, pvals, diffs, sesd = [], [], [], []
    for a, b in itertools.combinations(MODELS, 2):
        d, se = paired_diff(df, a, b)
        pairs.append(f"{a}-{b}"); diffs.append(100 * d); sesd.append(100 * se)
        pvals.append(float(2 * norm.sf(abs(d) / se)))
    pvals = np.array(pvals)
    num["pairwise"] = {p: {"diff_points": d, "se_points": s, "p": pv} for p, d, s, pv in zip(pairs, diffs, sesd, pvals)}
    num["n_tests"] = int(pvals.size)
    num["sig_raw"] = int((pvals <= 0.05).sum())
    num["sig_bonf"] = int(bonferroni(pvals).sum())
    num["sig_holm"] = int(holm(pvals).sum())
    num["sig_bh"] = int(benjamini_hochberg(pvals).sum())
    # cross-check with statsmodels
    for meth, key in (("bonferroni", "sig_bonf"), ("holm", "sig_holm"), ("fdr_bh", "sig_bh")):
        r = multipletests(pvals, alpha=0.05, method=meth)[0]
        assert int(r.sum()) == num[key], (meth, int(r.sum()), num[key])
    # the pairs that are NOT rejected under Holm (the indistinguishable ones)
    rej_holm = holm(pvals); rej_bh = benjamini_hochberg(pvals); rej_raw = pvals <= 0.05
    num["not_rejected_holm"] = [p for p, r in zip(pairs, rej_holm) if not r]
    num["not_rejected_bh"] = [p for p, r in zip(pairs, rej_bh) if not r]
    num["not_rejected_raw"] = [p for p, r in zip(pairs, rej_raw) if not r]
    # comparisons among the top four
    top4 = order[:4]
    num["top4"] = top4
    num["top4_pairs"] = {}
    for a, b in itertools.combinations(top4, 2):
        key = f"{a}-{b}" if f"{a}-{b}" in num["pairwise"] else f"{b}-{a}"
        num["top4_pairs"][key] = num["pairwise"][key]
    # comparisons of the top model with the rest, and the plausibly-best set under Holm
    top = order[0]
    top_p = {}
    for other in MODELS:
        if other == top:
            continue
        key = f"{top}-{other}" if f"{top}-{other}" in num["pairwise"] else f"{other}-{top}"
        top_p[other] = num["pairwise"][key]["p"]
    others = list(top_p)
    rej = holm(np.array([top_p[o] for o in others]))
    num["plausibly_best_holm"] = [top] + [o for o, r in zip(others, rej) if not r]
    num["top_vs_rest_p"] = top_p

    # per-benchmark pairwise tests: 66 x 6
    pb_p = []
    for bname, g in df.groupby("benchmark"):
        for a, b in itertools.combinations(MODELS, 2):
            w = (g[a] - g[b]).to_numpy(float)
            se = w.std(ddof=1) / math.sqrt(w.size)
            pb_p.append(2 * norm.sf(abs(w.mean()) / se) if se > 0 else 1.0)
    pb_p = np.array(pb_p)
    num["perbench_tests"] = int(pb_p.size)
    num["perbench_sig_raw"] = int((pb_p <= 0.05).sum())
    num["perbench_sig_holm"] = int(holm(pb_p).sum())
    num["perbench_sig_bh"] = int(benjamini_hochberg(pb_p).sum())

    # --- 6.4 winner's curse -------------------------------------------------------
    for m in (2, 3, 5, 10, 12, 20, 50, 100):
        num[f"emax_m{m}"] = expected_max_normal(m)
    num["emax_2_closed"] = 1 / math.sqrt(math.pi)
    # permutation null: shuffle model labels within each item -> all models equal
    rng = np.random.default_rng(6)
    X = df[MODELS].to_numpy()
    bench = df["benchmark"].to_numpy()
    bidx = {b: np.where(bench == b)[0] for b in sizes.index}
    grand = X.mean()
    reps = 2000
    max_excess, gap12, sig_raw_null, sig_holm_null = [], [], [], []
    for _ in range(reps):
        Xp = X.copy()
        for i in range(Xp.shape[0]):
            rng.shuffle(Xp[i])
        # macro scores
        sc = np.array([np.mean([Xp[bidx[b], j].mean() for b in bidx]) for j in range(12)])
        srt = np.sort(sc)[::-1]
        max_excess.append(srt[0] - sc.mean())
        gap12.append(srt[0] - srt[1])
        # how many of the 66 paired tests reject under the null, raw and Holm
        pv = []
        for a, b in itertools.combinations(range(12), 2):
            d, v = 0.0, 0.0
            for bb in bidx:
                w = Xp[bidx[bb], a] - Xp[bidx[bb], b]
                d += w.mean() / 6; v += w.var(ddof=1) / w.size / 36
            pv.append(2 * norm.sf(abs(d) / math.sqrt(v)) if v > 0 else 1.0)
        pv = np.array(pv)
        sig_raw_null.append(int((pv <= 0.05).sum()))
        sig_holm_null.append(int(holm(pv).sum()))
    max_excess, gap12 = np.array(max_excess), np.array(gap12)
    se_typ = np.mean([ses[k] for k in MODELS])
    num["null_se_typical_points"] = 100 * se_typ
    num["null_max_excess_points"] = 100 * float(max_excess.mean())
    num["null_max_excess_in_se"] = float(max_excess.mean() / se_typ)
    num["null_gap12_points"] = 100 * float(gap12.mean())
    num["null_gap12_q95_points"] = 100 * float(np.quantile(gap12, 0.95))
    num["null_sig_raw_mean"] = float(np.mean(sig_raw_null))
    num["null_any_raw_frac"] = float(np.mean(np.array(sig_raw_null) > 0))
    num["null_any_holm_frac"] = float(np.mean(np.array(sig_holm_null) > 0))
    num["null_reps"] = reps

    # --- 6.6 rank intervals by item bootstrap (within benchmark, all models together)
    rng = np.random.default_rng(66)
    B = 4000
    ranks = np.zeros((B, 12), int)
    boot_scores = np.zeros((B, 12))
    for r in range(B):
        sc = np.zeros(12)
        for b, idx in bidx.items():
            samp = idx[rng.integers(0, idx.size, idx.size)]
            sc += X[samp].mean(axis=0) / 6
        boot_scores[r] = sc
        ranks[r] = (-sc).argsort().argsort() + 1
    num["rank_boot_reps"] = B
    num["rank_intervals"] = {}
    for j, mdl in enumerate(MODELS):
        rk = ranks[:, j]
        num["rank_intervals"][mdl] = {
            "observed_rank": int(order.index(mdl) + 1),
            "lo": int(np.quantile(rk, 0.025)), "hi": int(np.quantile(rk, 0.975)),
            "p_first": float((rk == 1).mean()),
            "boot_lo_points": 100 * float(np.quantile(boot_scores[:, j], 0.025)),
            "boot_hi_points": 100 * float(np.quantile(boot_scores[:, j], 0.975)),
        }
    num["plausibly_best_boot"] = [m_ for m_ in MODELS if num["rank_intervals"][m_]["p_first"] > 0.025]
    num["p_first_top"] = num["rank_intervals"][order[0]]["p_first"]

    OUT.write_text(json.dumps(num, indent=2))
    for k, v in num.items():
        if isinstance(v, (int, float)):
            print(f"{k:30s} {v:.5f}")
    print("order:", order)
    print("plausibly best (Holm):", num["plausibly_best_holm"], " (bootstrap):", num["plausibly_best_boot"])
    for mdl in order:
        ri = num["rank_intervals"][mdl]
        print(f"  {mdl:4s} {num['scores'][mdl]:.2f} +- {num['ses_points'][mdl]:.2f}  rank {ri['observed_rank']:2d} [{ri['lo']}, {ri['hi']}]  P(first)={ri['p_first']:.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
