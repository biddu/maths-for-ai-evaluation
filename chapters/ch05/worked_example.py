"""Chapter 5 -- every number printed in the chapter, regenerated from data.

Run:  python chapters/ch05/worked_example.py
Writes chapters/ch05/numbers.json.
"""
import json
import math
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).with_name("numbers.json")
Z = norm.ppf(0.975)


# ---------------------------------------------------------------------------
# From-scratch functions (Sections 5.2-5.6)
# ---------------------------------------------------------------------------
def balanced_components(X):
    """X: n x k array of 0/1. One-way ANOVA variance components (Section 5.4)."""
    n, k = X.shape
    item_means = X.mean(axis=1)
    grand = X.mean()
    ms_between = k * ((item_means - grand) ** 2).sum() / (n - 1)
    ms_within = ((X - item_means[:, None]) ** 2).sum() / (n * (k - 1))
    sb2 = max((ms_between - ms_within) / k, 0.0)          # between-item variance
    sw2 = ms_within                                          # within-item variance
    icc = sb2 / (sb2 + sw2)
    return dict(n=n, k=k, grand=grand, ms_between=ms_between, ms_within=ms_within,
                sb2=sb2, sw2=sw2, icc=icc, deff=1 + (k - 1) * icc, n_eff=n * k / (1 + (k - 1) * icc))


def se_naive(X):
    p = X.mean()
    return math.sqrt(p * (1 - p) / X.size)


def se_cluster(X):
    """Treat each item's mean as one observation: SD of item means / sqrt(n)."""
    m = X.mean(axis=1)
    return m.std(ddof=1) / math.sqrt(m.size)


def se_ratio_cluster(totals, sizes):
    """Cluster-robust SE for a proportion with unequal clusters (eq. 5.14)."""
    C = totals.size
    M = sizes.sum()
    p = totals.sum() / M
    resid = totals - p * sizes
    return p, math.sqrt(C / (C - 1) * (resid ** 2).sum()) / M


def cluster_bootstrap(X, reps=10_000, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.integers(0, n, (reps, n))
    return X[idx].reshape(reps, -1).mean(axis=1)


def cluster_bootstrap_unequal(totals, sizes, reps=10_000, seed=0):
    rng = np.random.default_rng(seed)
    C = totals.size
    idx = rng.integers(0, C, (reps, C))
    return totals[idx].sum(axis=1) / sizes[idx].sum(axis=1)


def optimal_k(cost_item, cost_sample, icc):
    return math.sqrt(cost_item / cost_sample * (1 - icc) / icc)


def main():
    num = {"z": Z}

    # --- HumanEval, 164 items x 20 samples, models A and B ------------------------
    he = pd.read_csv(ROOT / "data" / "humaneval_k20.csv")
    XA = he[he.model == "A"].pivot(index="item", columns="sample", values="pass").to_numpy()
    XB = he[he.model == "B"].pivot(index="item", columns="sample", values="pass").to_numpy()
    for tag, X in (("A", XA), ("B", XB)):
        comp = balanced_components(X)
        num[f"{tag}_n"], num[f"{tag}_k"] = comp["n"], comp["k"]
        num[f"{tag}_p_all"] = comp["grand"]
        num[f"{tag}_ms_between"], num[f"{tag}_ms_within"] = comp["ms_between"], comp["ms_within"]
        num[f"{tag}_sb2"], num[f"{tag}_sw2"], num[f"{tag}_icc"] = comp["sb2"], comp["sw2"], comp["icc"]
        num[f"{tag}_deff"], num[f"{tag}_n_eff"] = comp["deff"], comp["n_eff"]
        num[f"{tag}_se_naive_points"] = 100 * se_naive(X)
        num[f"{tag}_se_cluster_points"] = 100 * se_cluster(X)
        num[f"{tag}_se_ratio"] = se_cluster(X) / se_naive(X)
        num[f"{tag}_p_total_var"] = comp["grand"] * (1 - comp["grand"])
        num[f"{tag}_var_check"] = comp["sb2"] + comp["sw2"]       # should be close to p(1-p)
        # item-mean spread
        m = X.mean(axis=1)
        num[f"{tag}_items_all_pass"] = int((m == 1).sum())
        num[f"{tag}_items_all_fail"] = int((m == 0).sum())
        num[f"{tag}_items_mixed"] = int(((m > 0) & (m < 1)).sum())
        num[f"{tag}_item_mean_sd"] = float(m.std(ddof=1))
        # single-sample SE (Chapter 1) for comparison
        p1 = X[:, 0].mean()
        num[f"{tag}_p_first"] = p1
        num[f"{tag}_se_first_points"] = 100 * math.sqrt(p1 * (1 - p1) / X.shape[0])
        # cluster bootstrap
        boots = cluster_bootstrap(X, seed=1 if tag == "A" else 2)
        num[f"{tag}_boot_se_points"] = 100 * float(boots.std(ddof=1))
        num[f"{tag}_boot_lo"], num[f"{tag}_boot_hi"] = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
        # SE against k, from the fitted components: Var = (sb2 + sw2/k)/n
        ks = [1, 2, 3, 5, 10, 20, 50, 100, 1000]
        num[f"{tag}_se_vs_k_points"] = {str(kk): 100 * math.sqrt((comp["sb2"] + comp["sw2"] / kk) / comp["n"]) for kk in ks}
        num[f"{tag}_se_limit_points"] = 100 * math.sqrt(comp["sb2"] / comp["n"])
        # what a naive analysis would claim at k = 20
        num[f"{tag}_se_naive_claims_n"] = comp["n"] * comp["k"]

    # Miller-style decomposition for model A: share of Var(p_hat) from items vs samples at k = 20
    cA = balanced_components(XA)
    num["A_var_share_between"] = cA["sb2"] / (cA["sb2"] + cA["sw2"] / cA["k"])
    num["A_var_share_within"] = 1 - num["A_var_share_between"]

    # --- paired comparison A vs B with 20 samples each ------------------------------
    dA, dB = XA.mean(axis=1), XB.mean(axis=1)
    d = dA - dB
    num["diff_mean_points"] = 100 * d.mean()
    num["diff_se_cluster_points"] = 100 * d.std(ddof=1) / math.sqrt(d.size)
    # naive: treat 3280 vs 3280 as independent Bernoullis (unpaired, unclustered)
    pA, pB = XA.mean(), XB.mean()
    num["diff_se_naive_points"] = 100 * math.sqrt((pA * (1 - pA) + pB * (1 - pB)) / XA.size)
    num["diff_t"] = d.mean() / (d.std(ddof=1) / math.sqrt(d.size))
    num["diff_p"] = float(2 * tdist.sf(abs(num["diff_t"]), d.size - 1))
    num["diff_ci_lo_points"] = 100 * (d.mean() - Z * d.std(ddof=1) / math.sqrt(d.size))
    num["diff_ci_hi_points"] = 100 * (d.mean() + Z * d.std(ddof=1) / math.sqrt(d.size))
    num["diff_corr_item_means"] = float(np.corrcoef(dA, dB)[0, 1])
    # compare with the single-sample paired analysis of Chapter 3 (SE 4.22 points)
    num["diff_se_single_sample_points"] = 100 * (XA[:, 0] - XB[:, 0]).std(ddof=1) / math.sqrt(XA.shape[0])

    # --- mixed model (statsmodels MixedLM, linear probability, random intercept per item)
    try:
        import statsmodels.formula.api as smf
        long = he[he.model == "A"][["item", "pass"]].rename(columns={"pass": "y"}).copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm("y ~ 1", long, groups=long["item"]).fit(reml=True)
        num["mixedlm_intercept"] = float(md.params["Intercept"])
        num["mixedlm_se_points"] = 100 * float(md.bse["Intercept"])
        num["mixedlm_sb2"] = float(md.cov_re.iloc[0, 0])
        num["mixedlm_sw2"] = float(md.scale)
        num["mixedlm_icc"] = num["mixedlm_sb2"] / (num["mixedlm_sb2"] + num["mixedlm_sw2"])
    except Exception as exc:  # pragma: no cover
        num["mixedlm_error"] = str(exc)

    # --- passages: unequal clusters, model H ------------------------------------------
    pq = pd.read_csv(ROOT / "data" / "cluster_passages.csv")
    g = pq.groupby("passage")["correct"]
    totals, sizes = g.sum().to_numpy(float), g.size().to_numpy(float)
    C, M = totals.size, sizes.sum()
    p, se_c = se_ratio_cluster(totals, sizes)
    num["H_passages"], num["H_questions"] = int(C), int(M)
    num["H_size_min"], num["H_size_max"], num["H_size_mean"] = int(sizes.min()), int(sizes.max()), float(sizes.mean())
    num["H_p"] = p
    num["H_se_naive_points"] = 100 * math.sqrt(p * (1 - p) / M)
    num["H_se_cluster_points"] = 100 * se_c
    num["H_se_ratio"] = se_c / math.sqrt(p * (1 - p) / M)
    num["H_deff_observed"] = num["H_se_ratio"] ** 2
    num["H_n_eff"] = M / num["H_deff_observed"]
    # ICC for unequal clusters by the ANOVA estimator (Section 5.7)
    y = pq["correct"].to_numpy(float)
    means = g.mean().to_numpy()
    ss_between = (sizes * (means - p) ** 2).sum()
    ss_within = ((y - np.repeat(means, sizes.astype(int))) ** 2).sum()
    ms_b = ss_between / (C - 1)
    ms_w = ss_within / (M - C)
    k0 = (M - (sizes ** 2).sum() / M) / (C - 1)
    sb2 = max((ms_b - ms_w) / k0, 0.0)
    num["H_k0"] = k0
    num["H_icc"] = sb2 / (sb2 + ms_w)
    num["H_deff_formula"] = 1 + (k0 - 1) * num["H_icc"]
    boots = cluster_bootstrap_unequal(totals, sizes, seed=3)
    num["H_boot_se_points"] = 100 * float(boots.std(ddof=1))
    num["H_boot_lo"], num["H_boot_hi"] = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    num["H_wilson_naive_lo"], num["H_wilson_naive_hi"] = (
        (p + Z * Z / (2 * M) - Z * math.sqrt(p * (1 - p) / M + Z * Z / (4 * M * M))) / (1 + Z * Z / M),
        (p + Z * Z / (2 * M) + Z * math.sqrt(p * (1 - p) / M + Z * Z / (4 * M * M))) / (1 + Z * Z / M))
    num["H_ci_cluster_lo"], num["H_ci_cluster_hi"] = p - Z * se_c, p + Z * se_c
    num["H_t59"] = float(tdist.ppf(0.975, C - 1))
    num["H_ci_cluster_t_lo"], num["H_ci_cluster_t_hi"] = p - num["H_t59"] * se_c, p + num["H_t59"] * se_c
    num["H_passages_all_correct"] = int((totals == sizes).sum())
    num["H_passages_all_wrong"] = int((totals == 0).sum())

    # --- stratified benchmark, model I (Section 5.9.3) ---------------------------
    st = pd.read_csv(ROOT / "data" / "strata_subjects.csv")
    gs = st.groupby("subject")["correct"]
    n_h = gs.size().to_numpy(float); p_h = gs.mean().to_numpy()
    N = n_h.sum(); W = n_h / N
    p_st = (W * p_h).sum()
    var_naive = p_st * (1 - p_st) / N
    var_strat = (W ** 2 * p_h * (1 - p_h) / n_h).sum()          # = (1/N) sum W_h p_h(1-p_h) under proportional allocation
    between = (W * (p_h - p_st) ** 2).sum()
    within = (W * p_h * (1 - p_h)).sum()
    num["I_subjects"], num["I_items"], num["I_per_subject"] = int(n_h.size), int(N), int(n_h[0])
    num["I_p"] = p_st
    num["I_subject_rates"] = {s_: float(r_) for s_, r_ in zip(gs.mean().index, p_h)}
    num["I_rate_min"], num["I_rate_max"] = float(p_h.min()), float(p_h.max())
    num["I_within"], num["I_between"], num["I_total_check"] = within, between, within + between
    num["I_p1p"] = p_st * (1 - p_st)
    num["I_se_naive_points"] = 100 * math.sqrt(var_naive)
    num["I_se_strat_points"] = 100 * math.sqrt(var_strat)
    num["I_deff"] = var_strat / var_naive
    num["I_se_ratio"] = math.sqrt(var_strat / var_naive)
    num["I_n_eff"] = N / num["I_deff"]
    # stratified bootstrap check: resample within each subject
    rng2 = np.random.default_rng(5)
    y_by = [gs.get_group(k).to_numpy() for k in gs.groups]
    boots = []
    for _ in range(10_000):
        boots.append(sum(y[rng2.integers(0, y.size, y.size)].sum() for y in y_by) / N)
    boots = np.array(boots)
    num["I_boot_strat_se_points"] = 100 * float(boots.std(ddof=1))
    # what an SRS bootstrap (ignoring strata) gives
    yall = st["correct"].to_numpy()
    idx = rng2.integers(0, yall.size, (10_000, yall.size))
    num["I_boot_srs_se_points"] = 100 * float(yall[idx].mean(axis=1).std(ddof=1))
    # how much the gain depends on spread: same within, between scaled
    num["I_deff_if_rates_within_10pts"] = 1 - 0.0025 / (num["I_p1p"])  # illustrative: spread sd 0.05

    # --- design: optimal samples per item (Section 5.8 / Exercise) ----------------------
    for ratio in (1, 5, 20):
        num[f"k_opt_cost{ratio}_iccA"] = optimal_k(ratio, 1, cA["icc"])
    num["k_opt_cost5_icc0.3"] = optimal_k(5, 1, 0.3)
    num["k_opt_cost5_icc0.1"] = optimal_k(5, 1, 0.1)
    # budget example: 3280 model calls; SE for (n, k) combinations under fitted components
    for n_, k_ in ((3280, 1), (1640, 2), (656, 5), (328, 10), (164, 20), (82, 40)):
        num[f"se_budget_n{n_}_k{k_}_points"] = 100 * math.sqrt((cA["sb2"] + cA["sw2"] / k_) / n_)

    # --- coverage simulation for figure 5.2 / exercise: naive vs cluster interval -------
    rng = np.random.default_rng(7)
    sim = {}
    for icc in (0.0, 0.1, 0.3, 0.5):
        for C_ in (10, 30, 100):
            k_ = 5
            cover_naive = cover_cluster = cover_cluster_t = 0
            reps = 4000
            tau2 = icc * 0.21                                 # p(1-p) at p = 0.7
            for _ in range(reps):
                # per-cluster success probability with the given ICC (beta on the logit-free scale)
                a_ = 0.7 * (0.21 / tau2 - 1) if tau2 > 0 else None
                pc = rng.beta(a_, a_ * 0.3 / 0.7, C_) if tau2 > 0 else np.full(C_, 0.7)
                Xs = (rng.random((C_, k_)) < pc[:, None]).astype(float)
                ph = Xs.mean()
                se_n = math.sqrt(ph * (1 - ph) / Xs.size) if 0 < ph < 1 else 0.0
                se_cl = Xs.mean(axis=1).std(ddof=1) / math.sqrt(C_)
                cover_naive += abs(ph - 0.7) <= Z * se_n
                cover_cluster += abs(ph - 0.7) <= Z * se_cl
                cover_cluster_t += abs(ph - 0.7) <= tdist.ppf(0.975, C_ - 1) * se_cl
            sim[f"icc{icc}_C{C_}"] = dict(naive=cover_naive / reps, cluster=cover_cluster / reps,
                                          cluster_t=cover_cluster_t / reps)
    num["coverage_sim"] = sim

    OUT.write_text(json.dumps(num, indent=2))
    for kk, v in num.items():
        if isinstance(v, (int, float)):
            print(f"{kk:32s} {v:.5f}")
    print("\ncoverage (naive / cluster z / cluster t):")
    for kk, v in sim.items():
        print(f"  {kk:14s} {v['naive']:.3f} / {v['cluster']:.3f} / {v['cluster_t']:.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
