"""Chapter 4 -- every number printed in the chapter, regenerated from data.

Run:  python chapters/ch04/worked_example.py
Writes chapters/ch04/numbers.json.
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist, wilcoxon, binomtest

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).with_name("numbers.json")
Z = norm.ppf(0.975)


# ---------------------------------------------------------------------------
# From-scratch functions
# ---------------------------------------------------------------------------
def mean_se(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1) / math.sqrt(x.size)


def t_interval(x, alpha=0.05):
    m, se = mean_se(x)
    q = tdist.ppf(1 - alpha / 2, x.size - 1)
    return m - q * se, m + q * se


def bootstrap(stat, data, reps=10_000, seed=0):
    """Nonparametric bootstrap of stat(data) where data is an array of items (rows)."""
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = data.shape[0]
    return np.array([stat(data[rng.integers(0, n, n)]) for _ in range(reps)])


def percentile_ci(boots, alpha=0.05):
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def bca_ci(stat, data, boots, alpha=0.05):
    """Bias-corrected and accelerated interval (Section 4.3)."""
    data = np.asarray(data)
    n = data.shape[0]
    theta = stat(data)
    # bias correction from the fraction of replicates below the estimate
    z0 = norm.ppf(np.mean(boots < theta))
    # acceleration from the jackknife
    jack = np.array([stat(np.delete(data, i, axis=0)) for i in range(n)])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6 * (((jm - jack) ** 2).sum()) ** 1.5
    a = num / den if den > 0 else 0.0
    def adj(alpha_level):
        z = norm.ppf(alpha_level)
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = adj(alpha / 2), adj(1 - alpha / 2)
    return float(np.quantile(boots, lo)), float(np.quantile(boots, hi)), float(z0), float(a), float(lo), float(hi)


def signed_rank(d):
    """Wilcoxon signed-rank statistic with zeros dropped and ties averaged (Section 4.4)."""
    d = np.asarray(d, float)
    d = d[d != 0]
    m = d.size
    ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy()
    w_plus = ranks[d > 0].sum()
    mean = m * (m + 1) / 4
    # tie correction to the variance
    _, counts = np.unique(np.abs(d), return_counts=True)
    var = m * (m + 1) * (2 * m + 1) / 24 - (counts ** 3 - counts).sum() / 48
    z = (w_plus - mean) / math.sqrt(var)
    return w_plus, m, mean, var, z, 2 * norm.sf(abs(z))


def brier_decomposition(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    n = conf.size
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ybar = correct.mean()
    rel = res = 0.0
    ece = 0.0
    rows = []
    for k in range(bins):
        sel = idx == k
        nk = int(sel.sum())
        if nk == 0:
            continue
        qk, yk = conf[sel].mean(), correct[sel].mean()
        rel += nk * (qk - yk) ** 2
        res += nk * (yk - ybar) ** 2
        ece += nk * abs(qk - yk)
        rows.append({"bin": k, "n": nk, "conf": qk, "acc": yk})
    brier = ((conf - correct) ** 2).mean()
    unc = ybar * (1 - ybar)
    return {"brier": brier, "reliability": rel / n, "resolution": res / n, "uncertainty": unc,
            "ece": ece / n, "bins": rows, "check": rel / n - res / n + unc}


def main():
    num = {"z": Z}
    m = pd.read_csv(ROOT / "data" / "mtbench_judge.csv")
    e = m[m.model == "E"].sort_values("question")["score"].to_numpy(float)
    f = m[m.model == "F"].sort_values("question")["score"].to_numpy(float)
    n = e.size
    num["n_questions"] = int(n)

    # --- 4.1 means, SEs, t-intervals -------------------------------------------
    for tag, x in (("E", e), ("F", f)):
        mu, se = mean_se(x)
        lo, hi = t_interval(x)
        num[f"{tag}_mean"], num[f"{tag}_sd"], num[f"{tag}_se"] = mu, x.std(ddof=1), se
        num[f"{tag}_t_lo"], num[f"{tag}_t_hi"] = lo, hi
        num[f"{tag}_frac10"] = float((x == 10).mean())
        num[f"{tag}_frac_ge8"] = float((x >= 8).mean())
        num[f"{tag}_median"] = float(np.median(x))
        num[f"{tag}_hist"] = np.bincount(x.astype(int), minlength=11)[1:].tolist()
    num["t_quantile_79"] = float(tdist.ppf(0.975, n - 1))
    num["popoviciu_max_sd_1to10"] = 4.5
    num["popoviciu_max_se_1to10_n80"] = 4.5 / math.sqrt(n)
    num["popoviciu_max_se_1to5_n80"] = 2.0 / math.sqrt(n)
    # unpaired vs paired difference
    d = e - f
    num["diff_mean"] = d.mean()
    num["diff_se_paired"] = d.std(ddof=1) / math.sqrt(n)
    num["diff_se_unpaired"] = math.sqrt(num["E_se"] ** 2 + num["F_se"] ** 2)
    num["diff_corr"] = float(np.corrcoef(e, f)[0, 1])
    num["diff_t_lo"], num["diff_t_hi"] = t_interval(d)
    num["diff_ties"] = int((d == 0).sum())
    num["diff_pos"], num["diff_neg"] = int((d > 0).sum()), int((d < 0).sum())
    num["diff_hist"] = {str(int(k)): int(v) for k, v in zip(*np.unique(d, return_counts=True))}

    # --- 4.2/4.3 bootstrap: percentile and BCa for the mean of E and for the difference
    boots_e = bootstrap(lambda x: x.mean(), e, seed=1)
    num["E_boot_se"] = float(boots_e.std(ddof=1))
    num["E_boot_pct_lo"], num["E_boot_pct_hi"] = percentile_ci(boots_e)
    lo, hi, z0, a, alo, ahi = bca_ci(lambda x: x.mean(), e, boots_e)
    num.update({"E_bca_lo": lo, "E_bca_hi": hi, "E_bca_z0": z0, "E_bca_a": a, "E_bca_alo": alo, "E_bca_ahi": ahi})
    pairs = np.column_stack([e, f])
    boots_d = bootstrap(lambda p: p[:, 0].mean() - p[:, 1].mean(), pairs, seed=2)
    num["diff_boot_se"] = float(boots_d.std(ddof=1))
    num["diff_boot_pct_lo"], num["diff_boot_pct_hi"] = percentile_ci(boots_d)
    lo, hi, z0, a, alo, ahi = bca_ci(lambda p: p[:, 0].mean() - p[:, 1].mean(), pairs, boots_d)
    num.update({"diff_bca_lo": lo, "diff_bca_hi": hi, "diff_bca_z0": z0, "diff_bca_a": a})
    # a skewed statistic where percentile and BCa differ: the fraction of 10s for model E
    stat10 = lambda x: (x == 10).mean()
    boots_10 = bootstrap(stat10, e, seed=3)
    num["E_frac10_boot_pct_lo"], num["E_frac10_boot_pct_hi"] = percentile_ci(boots_10)
    lo, hi, z0, a, alo, ahi = bca_ci(stat10, e, boots_10)
    num.update({"E_frac10_bca_lo": lo, "E_frac10_bca_hi": hi, "E_frac10_bca_z0": z0, "E_frac10_bca_a": a})
    # a really skewed one: the mean of the log-odds of confidence on the 40 hardest MMLU items (used in Exercise)
    np.save(pathlib.Path(__file__).with_name("boots_diff.npy"), boots_d)

    # --- 4.4 ordinal: sign test and Wilcoxon signed-rank --------------------------
    w_plus, m_nz, w_mean, w_var, w_z, w_p = signed_rank(d)
    num.update({"wilcoxon_wplus": float(w_plus), "wilcoxon_m": int(m_nz), "wilcoxon_mean": w_mean,
                "wilcoxon_var": w_var, "wilcoxon_z": w_z, "wilcoxon_p": w_p})
    r = wilcoxon(e, f, zero_method="wilcox", correction=False, method="approx")
    num["wilcoxon_scipy_stat"], num["wilcoxon_scipy_p"] = float(r.statistic), float(r.pvalue)
    num["sign_test_p"] = float(binomtest(num["diff_pos"], num["diff_pos"] + num["diff_neg"], 0.5).pvalue)
    num["paired_t_stat"] = num["diff_mean"] / num["diff_se_paired"]
    num["paired_t_p"] = float(2 * tdist.sf(abs(num["paired_t_stat"]), n - 1))
    # proportion scoring >= 8 as a binary summary, paired (McNemar) on the same questions
    b_ = int(((e >= 8) & (f < 8)).sum()); c_ = int(((e < 8) & (f >= 8)).sum())
    num["ge8_b"], num["ge8_c"] = b_, c_
    num["ge8_mcnemar_p"] = float(binomtest(b_, b_ + c_, 0.5).pvalue)

    # --- 4.5/4.6 proper scoring and calibration ------------------------------------
    g = pd.read_csv(ROOT / "data" / "mmlu_probs.csv")
    conf, correct = g["confidence"].to_numpy(float), g["correct"].to_numpy(float)
    num["mmlu_n"] = int(conf.size)
    num["mmlu_acc"] = correct.mean()
    num["mmlu_mean_conf"] = conf.mean()
    num["mmlu_brier"] = ((conf - correct) ** 2).mean()
    num["mmlu_logloss"] = float(-np.mean(correct * np.log(conf) + (1 - correct) * np.log(1 - conf)))
    num["mmlu_brier_constant"] = correct.mean() * (1 - correct.mean())   # predicting the base rate
    num["mmlu_brier_abs_error"] = float(np.mean(np.abs(conf - correct)))
    for bins in (5, 10, 15, 20, 50):
        dec = brier_decomposition(conf, correct, bins)
        num[f"mmlu_ece_{bins}"] = dec["ece"]
        num[f"mmlu_rel_{bins}"], num[f"mmlu_res_{bins}"] = dec["reliability"], dec["resolution"]
        if bins == 10:
            num["mmlu_unc"] = dec["uncertainty"]
            num["mmlu_decomp_check"] = dec["check"]
            num["mmlu_bins10"] = dec["bins"]
    num["mmlu_ece_range_pct"] = 100 * (max(num[f"mmlu_ece_{b}"] for b in (5, 10, 15, 20, 50)) /
                                       min(num[f"mmlu_ece_{b}"] for b in (5, 10, 15, 20, 50)) - 1)
    # Platt-style recalibration on the logit scale: q' = sigmoid(a * logit(q) + b), fitted by log loss
    lg = np.log(conf / (1 - conf))
    best = None
    for a_ in np.linspace(0.3, 1.2, 181):
        for b_ in np.linspace(-1.0, 0.5, 151):
            q = 1 / (1 + np.exp(-(a_ * lg + b_)))
            ll = -np.mean(correct * np.log(q) + (1 - correct) * np.log(1 - q))
            if best is None or ll < best[0]:
                best = (float(ll), float(a_), float(b_))
    num["recal_logloss"], num["recal_a"], num["recal_b"] = best
    qcal = 1 / (1 + np.exp(-(best[1] * lg + best[2])))
    num["recal_brier"] = float(((qcal - correct) ** 2).mean())
    num["recal_mean_conf"] = float(qcal.mean())
    for bins in (5, 10, 15, 20, 50):
        num[f"recal_ece_{bins}"] = brier_decomposition(qcal, correct, bins)["ece"]
    num["recal_ece_ratio_50_5"] = num["recal_ece_50"] / num["recal_ece_5"]
    num["recal_rel_10"] = brier_decomposition(qcal, correct, 10)["reliability"]
    np.save(pathlib.Path(__file__).with_name("qcal.npy"), qcal)
    bin_grid = list(range(2, 61))
    num["ece_bin_grid"] = bin_grid
    num["ece_curve_raw"] = [brier_decomposition(conf, correct, b)["ece"] for b in bin_grid]
    num["ece_curve_recal"] = [brier_decomposition(qcal, correct, b)["ece"] for b in bin_grid]
    num["recal_bins10"] = brier_decomposition(qcal, correct, 10)["bins"]
    # bootstrap interval for ECE(10) over items
    items = np.column_stack([conf, correct])
    boots_ece = bootstrap(lambda it: brier_decomposition(it[:, 0], it[:, 1], 10)["ece"], items, reps=2000, seed=4)
    num["mmlu_ece_10_boot_lo"], num["mmlu_ece_10_boot_hi"] = percentile_ci(boots_ece)

    OUT.write_text(json.dumps(num, indent=2))
    for k, v in num.items():
        if isinstance(v, (int, float)):
            print(f"{k:30s} {v:.5f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
