"""Chapter 3 -- every number printed in the chapter, regenerated from data.

Run:  python chapters/ch03/worked_example.py
Writes chapters/ch03/numbers.json.
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, norm
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).with_name("numbers.json")
Z = norm.ppf(0.975)


# ---------------------------------------------------------------------------
# From-scratch functions (Sections 3.2-3.5)
# ---------------------------------------------------------------------------
def concordance_table(x, y):
    """Counts (a, b, c, d): both pass, x only, y only, both fail."""
    x, y = np.asarray(x, int), np.asarray(y, int)
    a = int(((x == 1) & (y == 1)).sum())
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    d = int(((x == 0) & (y == 0)).sum())
    return a, b, c, d


def paired_diff_se(a, b, c, d):
    """Difference p_x - p_y and its paired standard error (eq. 3.9)."""
    n = a + b + c + d
    diff = (b - c) / n
    psi = (b + c) / n                      # discordance rate
    var = (psi - diff * diff) / n
    return diff, math.sqrt(var)


def unpaired_diff_se(a, b, c, d):
    n = a + b + c + d
    px, py = (a + b) / n, (a + c) / n
    return math.sqrt((px * (1 - px) + py * (1 - py)) / n)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value: b ~ Binomial(b + c, 1/2) under H0."""
    return binomtest(b, b + c, 0.5).pvalue


def mcnemar_chi2(b, c, correction=False):
    num = (abs(b - c) - 1) ** 2 if correction else (b - c) ** 2
    stat = num / (b + c)
    return stat, chi2.sf(stat, 1)


def paired_bootstrap(x, y, reps=10_000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, int), np.asarray(y, int)
    n = x.size
    idx = rng.integers(0, n, (reps, n))
    diffs = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    return diffs


def tango_interval(b, c, n, alpha=0.05, tol=1e-10):
    """Tango's score interval for delta = p_A - p_B (Section 3.5)."""
    z = norm.ppf(1 - alpha / 2)

    def score(delta):
        A = 2 * n
        B = -(b + c) + delta * (2 * n - b + c)
        C = -c * delta * (1 - delta)
        p01 = (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)   # restricted MLE of pi_01
        var = n * (2 * p01 + delta * (1 - delta))
        return (b - c - n * delta) / math.sqrt(var) if var > 0 else math.inf

    def solve(lo, hi, target):
        # score(delta) is decreasing in delta; bisect for score(delta) = target
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if score(mid) > target:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)

    d_hat = (b - c) / n
    lower = solve(-1 + 1e-12, d_hat, z)      # score = +z on the left of d_hat
    upper = solve(d_hat, 1 - 1e-12, -z)      # score = -z on the right
    return lower, upper, score


def noninferiority(b, c, n, margin, alpha=0.05):
    """One-sided non-inferiority test of delta > -margin (Section 3.8)."""
    z1 = norm.ppf(1 - alpha)
    diff, se = paired_diff_se(0, b, c, n - b - c)
    lower = diff - z1 * se
    return diff, se, lower, lower > -margin


def n_noninferiority(margin, true_delta, psi, power=0.8, alpha=0.05):
    z_a, z_b = norm.ppf(1 - alpha), norm.ppf(power)
    eff = true_delta + margin
    return math.ceil((z_a * math.sqrt(psi) + z_b * math.sqrt(psi - eff ** 2)) ** 2 / eff ** 2)


def power_mcnemar(n, delta, psi, alpha=0.05):
    """Normal-approximation power of the paired test (eq. 3.16)."""
    z = norm.ppf(1 - alpha / 2)
    return norm.cdf((math.sqrt(n) * abs(delta) - z * math.sqrt(psi)) / math.sqrt(psi - delta ** 2))


def n_for_power(delta, psi, power=0.8, alpha=0.05):
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return math.ceil((z_a * math.sqrt(psi) + z_b * math.sqrt(psi - delta ** 2)) ** 2 / delta ** 2)


def analyse(x, y, tag, num, boot_seed):
    a, b, c, d = concordance_table(x, y)
    n = a + b + c + d
    diff, se_p = paired_diff_se(a, b, c, d)
    se_u = unpaired_diff_se(a, b, c, d)
    px, py = (a + b) / n, (a + c) / n
    cov = a / n - px * py
    num.update({
        f"{tag}_n": n, f"{tag}_a": a, f"{tag}_b": b, f"{tag}_c": c, f"{tag}_d": d,
        f"{tag}_px": px, f"{tag}_py": py, f"{tag}_diff_points": 100 * diff,
        f"{tag}_psi": (b + c) / n, f"{tag}_cov": cov, f"{tag}_corr": cov / math.sqrt(px * (1 - px) * py * (1 - py)),
        f"{tag}_se_paired_points": 100 * se_p, f"{tag}_se_unpaired_points": 100 * se_u,
        f"{tag}_se_ratio": se_u / se_p,
        f"{tag}_z_paired": diff / se_p, f"{tag}_z_unpaired": diff / se_u,
        f"{tag}_p_wald_paired": 2 * norm.sf(abs(diff) / se_p),
        f"{tag}_p_wald_unpaired": 2 * norm.sf(abs(diff) / se_u),
        f"{tag}_ci_wald_lo_points": 100 * (diff - Z * se_p), f"{tag}_ci_wald_hi_points": 100 * (diff + Z * se_p),
        f"{tag}_ci_unpaired_lo_points": 100 * (diff - Z * se_u), f"{tag}_ci_unpaired_hi_points": 100 * (diff + Z * se_u),
        f"{tag}_mcnemar_exact_p": mcnemar_exact(b, c),
        f"{tag}_mcnemar_chi2": mcnemar_chi2(b, c)[0], f"{tag}_mcnemar_chi2_p": mcnemar_chi2(b, c)[1],
        f"{tag}_mcnemar_chi2cc": mcnemar_chi2(b, c, True)[0], f"{tag}_mcnemar_chi2cc_p": mcnemar_chi2(b, c, True)[1],
    })
    # statsmodels cross-check
    tbl = [[a, b], [c, d]]
    r_exact = sm_mcnemar(tbl, exact=True)
    r_asym = sm_mcnemar(tbl, exact=False, correction=True)
    assert abs(r_exact.pvalue - num[f"{tag}_mcnemar_exact_p"]) < 1e-9
    assert abs(r_asym.statistic - num[f"{tag}_mcnemar_chi2cc"]) < 1e-9
    # separate intervals for each model (Wilson), for the overlap discussion
    from statsmodels.stats.proportion import proportion_confint
    lo_x, hi_x = proportion_confint(a + b, n, method="wilson")
    lo_y, hi_y = proportion_confint(a + c, n, method="wilson")
    num.update({f"{tag}_wilson_x_lo": lo_x, f"{tag}_wilson_x_hi": hi_x,
                f"{tag}_wilson_y_lo": lo_y, f"{tag}_wilson_y_hi": hi_y,
                f"{tag}_intervals_overlap": bool(lo_x <= hi_y and lo_y <= hi_x)})
    # paired bootstrap
    diffs = paired_bootstrap(x, y, reps=10_000, seed=boot_seed)
    num.update({f"{tag}_boot_lo_points": 100 * float(np.quantile(diffs, 0.025)),
                f"{tag}_boot_hi_points": 100 * float(np.quantile(diffs, 0.975)),
                f"{tag}_boot_se_points": 100 * float(diffs.std(ddof=1)),
                f"{tag}_boot_reps": 10_000})
    return diffs


def main():
    num = {"z": Z}

    # --- HumanEval A vs B, first sample --------------------------------------
    he = pd.read_csv(ROOT / "data" / "humaneval_k20.csv")
    first = he[he["sample"] == 0]
    a_ = first[first.model == "A"].sort_values("item")["pass"].to_numpy()
    b_ = first[first.model == "B"].sort_values("item")["pass"].to_numpy()
    he_diffs = analyse(a_, b_, "he", num, boot_seed=1)

    # --- GSM8K C vs D ---------------------------------------------------------
    g = pd.read_csv(ROOT / "data" / "gsm8k_pairs.csv")
    g_diffs = analyse(g["C"].to_numpy(), g["D"].to_numpy(), "gsm", num, boot_seed=2)

    # --- the overlap rule (Section 3.6) --------------------------------------
    num["overlap_z_equal_se"] = 2 * Z / math.sqrt(2)          # 2.772
    num["overlap_p_equal_se"] = 2 * norm.sf(num["overlap_z_equal_se"])
    num["goldstein_healy_z"] = Z / math.sqrt(2)                 # 1.386
    num["goldstein_healy_level"] = 1 - 2 * norm.sf(num["goldstein_healy_z"])

    # --- power and sample size (Section 3.7) ----------------------------------
    for psi in (0.05, 0.10, 0.15, 0.20, 0.30):
        num[f"n_delta0.02_psi{psi}"] = n_for_power(0.02, psi)
        num[f"n_delta0.05_psi{psi}"] = n_for_power(0.05, psi)
    num["power_he_delta0.05_psi0.2"] = power_mcnemar(164, 0.05, 0.20)
    num["power_he_delta0.10_psi0.2"] = power_mcnemar(164, 0.10, 0.20)
    num["power_gsm_delta0.02_psi0.16"] = power_mcnemar(1319, 0.02, 0.16)
    num["power_gsm_delta0.02_psi0.30"] = power_mcnemar(1319, 0.02, 0.30)
    num["n_unpaired_delta0.02_p0.8"] = math.ceil((norm.ppf(0.975) + norm.ppf(0.8)) ** 2 * 2 * 0.8 * 0.2 / 0.02 ** 2)
    num["mdd_he_psi0.2_points"] = 100 * (Z * math.sqrt(0.2 / 164) + norm.ppf(0.8) * math.sqrt(0.2 / 164))
    # power-curve data for the figure: n needed vs psi at delta = 0.02, 0.03, 0.05
    grid = np.linspace(0.02, 0.5, 97)
    num["psi_grid"] = grid.tolist()
    for delta in (0.02, 0.03, 0.05):
        num[f"n_curve_delta{delta}"] = [n_for_power(delta, float(p)) if p > delta ** 2 else None for p in grid]

    # --- Tango score intervals (Section 3.5) ---------------------------------
    for tag in ("he", "gsm"):
        lo, hi, _ = tango_interval(num[f"{tag}_b"], num[f"{tag}_c"], num[f"{tag}_n"])
        num[f"{tag}_tango_lo_points"], num[f"{tag}_tango_hi_points"] = 100 * lo, 100 * hi
    # small-sample case for the text and Exercise 3.7: b = 6, c = 1, n = 30, and the table of Exercise 3.4
    for tag, (b_, c_, n_) in {"small": (6, 1, 30), "ex34": (12, 5, 164)}.items():
        d_, se_ = paired_diff_se(0, b_, c_, n_ - b_ - c_)
        lo, hi, _ = tango_interval(b_, c_, n_)
        num[f"{tag}_diff_points"] = 100 * d_
        num[f"{tag}_wald_lo_points"], num[f"{tag}_wald_hi_points"] = 100 * (d_ - Z * se_), 100 * (d_ + Z * se_)
        num[f"{tag}_tango_lo_points"], num[f"{tag}_tango_hi_points"] = 100 * lo, 100 * hi
        num[f"{tag}_mcnemar_exact_p"] = mcnemar_exact(b_, c_)
    # score curve for the figure: Z(delta) for the GSM8K table
    _, _, score = tango_interval(num["gsm_b"], num["gsm_c"], num["gsm_n"])
    dgrid = np.linspace(-0.02, 0.07, 361)
    num["score_grid_points"] = (100 * dgrid).tolist()
    num["score_curve_gsm"] = [score(float(d)) for d in dgrid]

    # --- non-inferiority gates (Section 3.8) ---------------------------------
    num["z_one_sided"] = norm.ppf(0.95)
    # new = D, old = C on GSM8K: is D no worse than C by more than the margin?
    for margin in (0.02, 0.03, 0.05):
        diff, se, lower, ok = noninferiority(num["gsm_c"], num["gsm_b"], num["gsm_n"], margin)
        num[f"gate_gsm_margin{margin}_lower_points"] = 100 * lower
        num[f"gate_gsm_margin{margin}_pass"] = bool(ok)
    num["gate_gsm_diff_points"] = 100 * diff
    num["gate_gsm_se_points"] = 100 * se
    # new = A, old = B on HumanEval
    for margin in (0.03, 0.05):
        diff, se, lower, ok = noninferiority(num["he_b"], num["he_c"], num["he_n"], margin)
        num[f"gate_he_margin{margin}_lower_points"] = 100 * lower
        num[f"gate_he_margin{margin}_pass"] = bool(ok)
    num["gate_he_lower_points"] = 100 * lower
    # the margin at which HumanEval's gate would pass
    num["gate_he_margin_needed_points"] = -100 * lower
    # sample sizes for a non-inferiority gate
    for psi in (0.10, 0.16, 0.30):
        num[f"n_ni_margin0.02_delta0_psi{psi}"] = n_noninferiority(0.02, 0.0, psi)
        num[f"n_ni_margin0.02_delta-0.01_psi{psi}"] = n_noninferiority(0.02, -0.01, psi)
    num["n_ni_margin0.05_delta0_psi0.2"] = n_noninferiority(0.05, 0.0, 0.2)
    num["n_ni_margin0.03_delta0_psi0.2"] = n_noninferiority(0.03, 0.0, 0.2)

    # --- simulation for figure 3.1: sampling distribution of D, paired vs independent
    rng = np.random.default_rng(3)
    n = 164
    # population pair probabilities estimated from the HumanEval table
    pi = np.array([num["he_a"], num["he_b"], num["he_c"], num["he_d"]]) / n
    draws = rng.multinomial(n, pi, 20_000)
    num["sim_paired_diff_sd_points"] = 100 * float(((draws[:, 1] - draws[:, 2]) / n).std())
    px, py = num["he_px"], num["he_py"]
    indep = (rng.binomial(n, px, 20_000) - rng.binomial(n, py, 20_000)) / n
    num["sim_indep_diff_sd_points"] = 100 * float(indep.std())
    np.save(pathlib.Path(__file__).with_name("sim_paired.npy"), (draws[:, 1] - draws[:, 2]) / n)
    np.save(pathlib.Path(__file__).with_name("sim_indep.npy"), indep)

    OUT.write_text(json.dumps(num, indent=2))
    for k, v in num.items():
        if isinstance(v, (int, float, bool)):
            print(f"{k:34s} {v}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
