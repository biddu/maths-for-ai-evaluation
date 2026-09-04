"""Chapter 7 -- every number printed in the chapter, regenerated from data.

Run:  python chapters/ch07/worked_example.py
Writes chapters/ch07/numbers.json.

Data: data/raters300.csv (three human raters R1, R2, R3 on 300 items) and
data/judge_labels.csv (model J against an adjudicated human label, 600 items).
"""
import itertools
import json
import math
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).with_name("numbers.json")
Z = norm.ppf(0.975)
RATERS = ["R1", "R2", "R3"]


# ---------------------------------------------------------------------------
# Two raters: the agreement table and Cohen's kappa (Sections 7.2-7.3)
# ---------------------------------------------------------------------------
def agreement_table(a, b, categories):
    """K x K table of counts; rows = rater a, columns = rater b."""
    K = len(categories)
    idx = {c: i for i, c in enumerate(categories)}
    T = np.zeros((K, K))
    for x, y in zip(a, b):
        T[idx[x], idx[y]] += 1
    return T


def weight_matrix(K, kind=None):
    """Agreement weights w_ij in [0, 1], 1 on the diagonal."""
    i, j = np.indices((K, K))
    if kind is None:
        return (i == j).astype(float)
    if kind == "linear":
        return 1 - np.abs(i - j) / (K - 1)
    if kind == "quadratic":
        return 1 - ((i - j) / (K - 1)) ** 2
    raise ValueError(kind)


def cohen_kappa(T, kind=None):
    """(kappa, p_o, p_e) from a K x K count table with optional weights."""
    P = T / T.sum()
    W = weight_matrix(P.shape[0], kind)
    r, c = P.sum(1), P.sum(0)
    po = (W * P).sum()
    pe = (W * np.outer(r, c)).sum()
    return (po - pe) / (1 - pe), po, pe


def kappa_se(T, kind=None):
    """Large-sample SE of (weighted) kappa, Fleiss, Cohen and Everitt (1969)."""
    n = T.sum()
    P = T / n
    W = weight_matrix(P.shape[0], kind)
    r, c = P.sum(1), P.sum(0)
    po = (W * P).sum()
    pe = (W * np.outer(r, c)).sum()
    wbar_row = W @ c            # \bar w_{i.} = sum_j p_{.j} w_ij
    wbar_col = W.T @ r          # \bar w_{.j} = sum_i p_{i.} w_ij
    core = (W * (1 - pe) - (wbar_row[:, None] + wbar_col[None, :]) * (1 - po)) ** 2
    var = ((P * core).sum() - (po * pe - 2 * pe + po) ** 2) / (n * (1 - pe) ** 4)
    return math.sqrt(var)


def kappa_se_null(T):
    """SE of unweighted kappa under the hypothesis kappa = 0 (for a test)."""
    n = T.sum()
    P = T / n
    r, c = P.sum(1), P.sum(0)
    pe = (r * c).sum()
    var = (pe + pe ** 2 - (r * c * (r + c)).sum()) / (n * (1 - pe) ** 2)
    return math.sqrt(var)


def kappa_max(T):
    """Largest kappa attainable with the observed marginals."""
    P = T / T.sum()
    r, c = P.sum(1), P.sum(0)
    po_max = np.minimum(r, c).sum()
    pe = (r * c).sum()
    return (po_max - pe) / (1 - pe), po_max


def binary_indices(T):
    """PABAK, prevalence index and bias index for a 2 x 2 table (Byrt et al. 1993)."""
    P = T / T.sum()
    a, b, c, d = P[1, 1], P[1, 0], P[0, 1], P[0, 0]      # 1 = positive
    po = a + d
    return {"pabak": 2 * po - 1, "PI": a - d, "BI": b - c}


def kappa_from_indices(pabak, PI, BI):
    return (pabak - PI ** 2 + BI ** 2) / (1 - PI ** 2 + BI ** 2)


def concordance_cc(x, y):
    """Lin's concordance correlation coefficient, population moments."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return 2 * np.mean((x - x.mean()) * (y - y.mean())) / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)


# ---------------------------------------------------------------------------
# Many raters: Fleiss' kappa and Krippendorff's alpha (Section 7.4)
# ---------------------------------------------------------------------------
def fleiss_kappa(counts):
    """counts: items x categories, number of raters choosing each category."""
    counts = np.asarray(counts, float)
    n = counts.sum(1)
    assert np.all(n == n[0]), "Fleiss' kappa needs the same number of raters per item"
    m = n[0]
    P_i = ((counts ** 2).sum(1) - m) / (m * (m - 1))      # per-item agreement
    Po = P_i.mean()
    pj = counts.sum(0) / counts.sum()
    Pe = (pj ** 2).sum()
    return (Po - Pe) / (1 - Pe), Po, Pe


def krippendorff_alpha(data, level="nominal", values=None):
    """data: raters x items array with np.nan for missing. Coincidence-matrix form."""
    data = np.asarray(data, float)
    if values is None:
        values = np.unique(data[~np.isnan(data)])
    values = np.asarray(values, float)
    K = values.size
    idx = {v: k for k, v in enumerate(values)}
    O = np.zeros((K, K))
    for u in range(data.shape[1]):                           # each item ("unit")
        col = data[:, u]
        col = col[~np.isnan(col)]
        m_u = col.size
        if m_u < 2:
            continue
        for a in col:
            for b in col:
                O[idx[a], idx[b]] += 1 / (m_u - 1)
        for a in col:                                        # remove the a == b self pairs
            O[idx[a], idx[a]] -= 1 / (m_u - 1)
    n_c = O.sum(1)
    n = n_c.sum()
    if level == "nominal":
        D = 1 - np.eye(K)
    elif level == "interval":
        D = (values[:, None] - values[None, :]) ** 2
    elif level == "ordinal":
        # distance between ranks g < h: (sum of n_k from g to h - (n_g + n_h)/2)^2
        D = np.zeros((K, K))
        for g in range(K):
            for h in range(K):
                lo, hi = min(g, h), max(g, h)
                D[g, h] = (n_c[lo:hi + 1].sum() - (n_c[g] + n_c[h]) / 2) ** 2
    else:
        raise ValueError(level)
    Do = (O * D).sum() / n
    De = (np.outer(n_c, n_c) * D).sum() / (n * (n - 1))
    return 1 - Do / De


# ---------------------------------------------------------------------------
# ICC from the two-way ANOVA (Section 7.5)
# ---------------------------------------------------------------------------
def icc_two_way(Y):
    """Y: items x raters, complete. Returns dict of ICC forms (McGraw & Wong 1996)."""
    Y = np.asarray(Y, float)
    n, k = Y.shape
    grand = Y.mean()
    MSR = k * ((Y.mean(1) - grand) ** 2).sum() / (n - 1)        # between items (rows)
    MSC = n * ((Y.mean(0) - grand) ** 2).sum() / (k - 1)        # between raters (columns)
    SSE = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + grand) ** 2).sum()
    MSE = SSE / ((n - 1) * (k - 1))
    MSW = ((Y - Y.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))   # one-way within
    out = {
        "MSR": MSR, "MSC": MSC, "MSE": MSE, "MSW": MSW,
        "ICC1": (MSR - MSW) / (MSR + (k - 1) * MSW),
        "ICC_A1": (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n),
        "ICC_C1": (MSR - MSE) / (MSR + (k - 1) * MSE),
    }
    out["ICC1k"] = (MSR - MSW) / MSR
    out["ICC_Ak"] = (MSR - MSE) / (MSR + (MSC - MSE) / n)
    out["ICC_Ck"] = (MSR - MSE) / MSR
    return out


def spearman_brown(rho1, k):
    return k * rho1 / (1 + (k - 1) * rho1)


# ---------------------------------------------------------------------------
def bootstrap_kappa(a, b, categories, kind=None, reps=4000, seed=7):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    n = a.size
    ks = np.empty(reps)
    for r in range(reps):
        s = rng.integers(0, n, n)
        ks[r] = cohen_kappa(agreement_table(a[s], b[s], categories), kind)[0]
    return ks


def main():
    num = {"z": Z}
    d = pd.read_csv(ROOT / "data" / "raters300.csv")
    flag = d.pivot(index="item", columns="rater", values="flag")
    qual = d.pivot(index="item", columns="rater", values="quality")
    n = flag.shape[0]
    num["n_items"] = int(n)
    num["n_raters"] = 3
    num["flag_rate"] = {r: float(flag[r].mean()) for r in RATERS}
    num["flag_count"] = {r: int(flag[r].sum()) for r in RATERS}
    num["quality_mean"] = {r: float(qual[r].mean()) for r in RATERS}
    num["quality_missing_R3"] = int(qual["R3"].isna().sum())

    # --- 7.1 the failure-box arithmetic ------------------------------------------
    num["fb_pe"] = 0.92 ** 2 + 0.08 ** 2
    num["fb_kappa"] = (0.92 - num["fb_pe"]) / (1 - num["fb_pe"])
    num["fb_pe_balanced"] = 0.5
    num["fb_kappa_balanced"] = (0.92 - 0.5) / 0.5

    # --- 7.2 pairwise Cohen's kappa on the binary flag ------------------------------
    from sklearn.metrics import cohen_kappa_score
    from statsmodels.stats.inter_rater import cohens_kappa as sm_kappa, fleiss_kappa as sm_fleiss
    num["pairs"] = {}
    for a, b in itertools.combinations(RATERS, 2):
        T = agreement_table(flag[a], flag[b], [0, 1])
        k, po, pe = cohen_kappa(T)
        se = kappa_se(T)
        kmax, pomax = kappa_max(T)
        ind = binary_indices(T)
        res = sm_kappa(T, return_results=True)
        assert abs(res.kappa - k) < 1e-12
        assert abs(math.sqrt(res.var_kappa) - se) < 1e-9, (math.sqrt(res.var_kappa), se)
        assert abs(math.sqrt(res.var_kappa0) - kappa_se_null(T)) < 1e-9
        assert abs(cohen_kappa_score(flag[a], flag[b]) - k) < 1e-12
        assert abs(kappa_from_indices(**ind) - k) < 1e-12
        boot = bootstrap_kappa(flag[a].to_numpy(), flag[b].to_numpy(), [0, 1])
        num["pairs"][f"{a}-{b}"] = {
            "table": T.astype(int).tolist(),        # rows a: [0,1]; cols b: [0,1]
            "po": po, "pe": pe, "kappa": k, "se": se, "se_null": kappa_se_null(T),
            "lo": k - Z * se, "hi": k + Z * se,
            "boot_se": float(boot.std(ddof=1)), "boot_lo": float(np.quantile(boot, 0.025)),
            "boot_hi": float(np.quantile(boot, 0.975)),
            "kappa_max": kmax, "po_max": pomax, **ind,
            "kappa_over_max": k / kmax,
        }

    # --- 7.3 the prevalence paradox: a rebalanced subset ---------------------------------
    # Keep every item R1 flagged, and a random 40 of the items R1 did not flag.
    rng = np.random.default_rng(73)
    pos = flag.index[flag["R1"] == 1].to_numpy()
    neg = flag.index[flag["R1"] == 0].to_numpy()
    sub = np.sort(np.concatenate([pos, rng.choice(neg, 40, replace=False)]))
    Tfull = agreement_table(flag.loc[:, "R2"], flag.loc[:, "R3"], [0, 1])
    Tsub = agreement_table(flag.loc[sub, "R2"], flag.loc[sub, "R3"], [0, 1])
    kf, pof, pef = cohen_kappa(Tfull)
    ks, pos_, pes = cohen_kappa(Tsub)
    num["rebalance"] = {
        "n_sub": int(sub.size), "n_pos_R1": int(pos.size),
        "prev_full": float(flag[["R2", "R3"]].to_numpy().mean()),
        "prev_sub": float(flag.loc[sub, ["R2", "R3"]].to_numpy().mean()),
        "po_full": pof, "pe_full": pef, "kappa_full": kf, "se_full": kappa_se(Tfull),
        "po_sub": pos_, "pe_sub": pes, "kappa_sub": ks, "se_sub": kappa_se(Tsub),
        "table_sub": Tsub.astype(int).tolist(),
        "pabak_full": binary_indices(Tfull)["pabak"], "pabak_sub": binary_indices(Tsub)["pabak"],
        "PI_full": binary_indices(Tfull)["PI"], "PI_sub": binary_indices(Tsub)["PI"],
    }
    # the same raters at a range of prevalences (analytic, R2-R3 sensitivities/specificities)
    # kappa for two raters with sens s_a, s_b and spec t_a, t_b, independent given truth
    def kappa_prev(prev, sa, sb, ta, tb):
        a = prev * sa * sb + (1 - prev) * (1 - ta) * (1 - tb)          # both positive
        d = prev * (1 - sa) * (1 - sb) + (1 - prev) * ta * tb          # both negative
        pa = prev * sa + (1 - prev) * (1 - ta)
        pb = prev * sb + (1 - prev) * (1 - tb)
        po = a + d
        pe = pa * pb + (1 - pa) * (1 - pb)
        return (po - pe) / (1 - pe), po
    num["kappa_prev_curve"] = {
        str(p): kappa_prev(p, 0.75, 0.70, 0.96, 0.95) for p in (0.02, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5)
    }

    # --- 7.3 weighted kappa on quality, and the concordance identity ------------------
    qc = qual.dropna()
    num["n_complete"] = int(qc.shape[0])
    cats = [1, 2, 3, 4, 5]
    num["quality_pairs"] = {}
    for a, b in itertools.combinations(RATERS, 2):
        T = agreement_table(qc[a].astype(int), qc[b].astype(int), cats)
        k0, po0, pe0 = cohen_kappa(T)
        kl = cohen_kappa(T, "linear")[0]
        kq, poq, peq = cohen_kappa(T, "quadratic")
        seq = kappa_se(T, "quadratic")
        ccc = concordance_cc(qc[a], qc[b])
        assert abs(kq - ccc) < 1e-12, (kq, ccc)
        Wq = weight_matrix(5, "quadratic")
        res = sm_kappa(T, weights=1 - Wq, return_results=True)   # statsmodels takes disagreement weights
        assert abs(res.kappa - kq) < 1e-12
        assert abs(math.sqrt(res.var_kappa) - seq) < 1e-9, (math.sqrt(res.var_kappa), seq)
        assert abs(cohen_kappa_score(qc[a], qc[b], weights="quadratic") - kq) < 1e-12
        assert abs(cohen_kappa_score(qc[a], qc[b], weights="linear") - kl) < 1e-12
        icc = icc_two_way(qc[[a, b]].to_numpy())
        num["quality_pairs"][f"{a}-{b}"] = {
            "po": po0, "kappa": k0, "kappa_linear": kl, "kappa_quadratic": kq, "se_quadratic": seq,
            "ccc": ccc, "icc_A1": icc["ICC_A1"], "icc_C1": icc["ICC_C1"],
            "pearson": float(np.corrcoef(qc[a], qc[b])[0, 1]),
            "mean_a": float(qc[a].mean()), "mean_b": float(qc[b].mean()),
            "within_one": float((np.abs(qc[a] - qc[b]) <= 1).mean()),
        }

    # --- 7.4 Fleiss' kappa and Krippendorff's alpha ---------------------------------------
    import krippendorff
    counts = np.stack([(flag == 0).sum(axis=1), (flag == 1).sum(axis=1)], axis=1)
    kf, Po, Pe = fleiss_kappa(counts)
    assert abs(sm_fleiss(counts) - kf) < 1e-12
    num["fleiss_flag"] = {"kappa": kf, "Po": Po, "Pe": Pe,
                          "pooled_rate": float(counts[:, 1].sum() / counts.sum())}
    # mean pairwise Cohen for comparison
    num["mean_pairwise_kappa_flag"] = float(np.mean([v["kappa"] for v in num["pairs"].values()]))
    # alpha, nominal, on the flag (no missing)
    R = flag[RATERS].to_numpy().T.astype(float)
    a_nom = krippendorff_alpha(R, "nominal")
    assert abs(krippendorff.alpha(reliability_data=R, level_of_measurement="nominal") - a_nom) < 1e-10
    num["alpha_flag_nominal"] = a_nom
    # Scott's pi for the pair R1-R2, and alpha for that pair only, to show the n/(n-1) link
    T12 = agreement_table(flag["R1"], flag["R2"], [0, 1])
    P12 = T12 / T12.sum()
    pooled = (P12.sum(0) + P12.sum(1)) / 2
    po12 = np.trace(P12); pe_scott = (pooled ** 2).sum()
    num["scott_pi_R1R2"] = (po12 - pe_scott) / (1 - pe_scott)
    num["alpha_R1R2_nominal"] = krippendorff_alpha(flag[["R1", "R2"]].to_numpy().T.astype(float), "nominal")
    N2 = 2 * n
    # exact relation for two raters, no missing: alpha = (pi (N-1) + 1)/N with N = 2n values
    num["alpha_pi_check"] = {"alpha": num["alpha_R1R2_nominal"], "pi": num["scott_pi_R1R2"],
                             "pi_plus": (num["scott_pi_R1R2"] * (N2 - 1) + 1) / N2}
    # quality: alpha at three levels with R3's missing cells kept
    Q = qual[RATERS].to_numpy().T.astype(float)
    num["alpha_quality"] = {}
    for lvl in ("nominal", "ordinal", "interval"):
        a_ = krippendorff_alpha(Q, lvl, values=cats)
        ref = krippendorff.alpha(reliability_data=Q, level_of_measurement=lvl, value_domain=cats)
        assert abs(a_ - ref) < 1e-10, (lvl, a_, ref)
        num["alpha_quality"][lvl] = a_
    # same on complete cases only, for comparison with Fleiss on quality
    Qc = qc[RATERS].to_numpy().T.astype(float)
    num["alpha_quality_complete"] = {lvl: krippendorff_alpha(Qc, lvl, values=cats) for lvl in ("nominal", "interval")}
    cq = np.stack([(qc == c).sum(axis=1) for c in cats], axis=1)
    num["fleiss_quality_complete"] = fleiss_kappa(cq)[0]
    # interval alpha equals ICC(A,1)-like agreement: compare
    num["n_pairable_values"] = int(np.sum(~np.isnan(Q)))

    # --- 7.5 ICC on quality (complete cases, three raters) -------------------------------
    icc = icc_two_way(qc[RATERS].to_numpy())
    num["icc"] = icc
    num["icc_sb_check"] = spearman_brown(icc["ICC_C1"], 3)
    assert abs(num["icc_sb_check"] - icc["ICC_Ck"]) < 1e-12
    # cross-check against pingouin
    import pingouin as pg
    long = qc.reset_index().melt(id_vars="item", var_name="rater", value_name="q")
    tab = pg.intraclass_corr(data=long, targets="item", raters="rater", ratings="q").set_index("Type")
    for mine, theirs in (("ICC1", "ICC(1,1)"), ("ICC_A1", "ICC(A,1)"), ("ICC_C1", "ICC(C,1)"),
                         ("ICC1k", "ICC(1,k)"), ("ICC_Ak", "ICC(A,k)"), ("ICC_Ck", "ICC(C,k)")):
        assert abs(tab.loc[theirs, "ICC"] - icc[mine]) < 1e-9, (mine, tab.loc[theirs, "ICC"], icc[mine])
    ci_col = [c for c in tab.columns if c.startswith("CI95")][0]
    num["icc_ci_A1"] = [float(x) for x in tab.loc["ICC(A,1)", ci_col]]
    num["icc_ci_C1"] = [float(x) for x in tab.loc["ICC(C,1)", ci_col]]
    # variance components implied by the two-way model
    k = 3
    s2_e = icc["MSE"]
    s2_item = (icc["MSR"] - icc["MSE"]) / k
    s2_rater = (icc["MSC"] - icc["MSE"]) / qc.shape[0]
    num["icc_components"] = {"item": s2_item, "rater": s2_rater, "error": s2_e}
    num["rater_sd_of_means"] = float(qc[RATERS].mean().std(ddof=1))

    # --- 7.6 precision and sample size -----------------------------------------------
    # planning with the R1-R2 table shape: SE scales as 1/sqrt(n)
    T = agreement_table(flag["R1"], flag["R2"], [0, 1])
    se300 = kappa_se(T)
    num["plan"] = {}
    for h in (0.10, 0.05):
        n_needed = (Z * se300 * math.sqrt(n) / h) ** 2
        num["plan"][f"halfwidth_{h}"] = n_needed
    # naive null-based planning number for comparison
    pe = num["pairs"]["R1-R2"]["pe"]
    num["plan"]["null_se_300"] = math.sqrt(pe / (n * (1 - pe)))
    # simulation check of the FCE standard error: same raters, same prevalence, 300 items
    rng = np.random.default_rng(76)
    reps = 4000
    ks = np.empty(reps); ses = np.empty(reps); cover = 0
    prev, sa, sb, ta, tb = 0.08, 0.80, 0.75, 0.97, 0.96
    ktrue = kappa_prev(prev, sa, sb, ta, tb)[0]
    for r in range(reps):
        t = rng.random(n) < prev
        u1, u2 = rng.random(n), rng.random(n)
        x = np.where(t, u1 < sa, u1 >= ta).astype(int)
        y = np.where(t, u2 < sb, u2 >= tb).astype(int)
        Tr = agreement_table(x, y, [0, 1])
        ks[r], _, _ = cohen_kappa(Tr)
        ses[r] = kappa_se(Tr)
        cover += (ks[r] - Z * ses[r] <= ktrue <= ks[r] + Z * ses[r])
    num["sim"] = {"kappa_true": ktrue, "sd_kappa": float(ks.std(ddof=1)), "mean_se": float(ses.mean()),
                  "coverage": cover / reps, "reps": reps}

    # --- 7.7 the judge: exact match against kappa ------------------------------------
    j = pd.read_csv(ROOT / "data" / "judge_labels.csv")
    T = agreement_table(j["human"], j["J"], [0, 1])
    kj, poj, pej = cohen_kappa(T)
    num["judge"] = {
        "n": int(j.shape[0]), "table": T.astype(int).tolist(),
        "prevalence": float(j["human"].mean()), "judge_rate": float(j["J"].mean()),
        "exact_match": poj, "pe": pej, "kappa": kj, "se": kappa_se(T),
        "gap_points": 100 * (poj - kj), "kappa_max": kappa_max(T)[0],
        "sens": float(j.loc[j.human == 1, "J"].mean()), "spec": float(1 - j.loc[j.human == 0, "J"].mean()),
        "pabak": binary_indices(T)["pabak"],
    }
    # the curve of exact match and kappa for a judge with these sens/spec across prevalence
    s, t = num["judge"]["sens"], num["judge"]["spec"]
    num["judge_curve"] = {}
    for p in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        a = p * s; dd = (1 - p) * t
        po = a + dd
        pj_ = p * s + (1 - p) * (1 - t)
        pe = pj_ * p + (1 - pj_) * (1 - p)
        num["judge_curve"][str(p)] = {"match": po, "kappa": (po - pe) / (1 - pe)}

    OUT.write_text(json.dumps(num, indent=2))
    for k_, v in num.items():
        if isinstance(v, (int, float)):
            print(f"{k_:28s} {v:.5f}")
    for name in ("pairs", "quality_pairs"):
        for p, v in num[name].items():
            print(name, p, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "table"})
    for name in ("rebalance", "fleiss_flag", "alpha_quality", "alpha_quality_complete", "alpha_pi_check", "icc", "icc_components", "plan", "sim", "judge", "kappa_prev_curve", "judge_curve"):
        print(name, num[name])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
