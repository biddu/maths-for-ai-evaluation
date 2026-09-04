"""Chapter 2 -- every number printed in the chapter, regenerated from scratch.

Run:  python chapters/ch02/worked_example.py
Writes chapters/ch02/numbers.json.
"""
import json
import math
import pathlib

import numpy as np
from scipy.stats import beta, binom, norm
from statsmodels.stats.proportion import proportion_confint

OUT = pathlib.Path(__file__).with_name("numbers.json")
Z = norm.ppf(0.975)


# ---------------------------------------------------------------------------
# The five intervals, from scratch (Sections 2.2-2.6)
# ---------------------------------------------------------------------------
def wald(s, n, z=Z):
    p = s / n
    h = z * math.sqrt(p * (1 - p) / n)
    return p - h, p + h


def wilson(s, n, z=Z):
    p = s / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def clopper_pearson(s, n, alpha=0.05):
    lo = 0.0 if s == 0 else beta.ppf(alpha / 2, s, n - s + 1)
    hi = 1.0 if s == n else beta.ppf(1 - alpha / 2, s + 1, n - s)
    return lo, hi


def agresti_coull(s, n, z=Z):
    n_t = n + z * z
    p_t = (s + z * z / 2) / n_t
    h = z * math.sqrt(p_t * (1 - p_t) / n_t)
    return p_t - h, p_t + h


def jeffreys(s, n, alpha=0.05):
    return beta.ppf(alpha / 2, s + 0.5, n - s + 0.5), beta.ppf(1 - alpha / 2, s + 0.5, n - s + 0.5)


METHODS = {"wald": wald, "wilson": wilson, "clopper_pearson": clopper_pearson,
           "agresti_coull": agresti_coull, "jeffreys": jeffreys}
SM_NAMES = {"wald": "normal", "wilson": "wilson", "clopper_pearson": "beta",
            "agresti_coull": "agresti_coull", "jeffreys": "jeffreys"}


def coverage(method, n, p, alpha=0.05):
    """Exact coverage probability of a 95% interval at population rate p (Section 2.7)."""
    total = 0.0
    for s in range(n + 1):
        lo, hi = method(s, n)
        if lo <= p <= hi:
            total += binom.pmf(s, n, p)
    return total


def main():
    num = {}

    # --- the intervals for the running cases -------------------------------
    cases = {"49of50": (49, 50), "117of164": (117, 164), "0of30": (0, 30),
             "12of12": (12, 12), "5of10": (5, 10)}
    for name, (s, n) in cases.items():
        for m, f in METHODS.items():
            lo, hi = f(s, n)
            num[f"{name}_{m}_lo"], num[f"{name}_{m}_hi"] = lo, hi
            # verify against statsmodels where the method exists
            slo, shi = proportion_confint(s, n, alpha=0.05, method=SM_NAMES[m])
            # statsmodels clips the Wald interval to [0, 1]; compare after clipping
            assert abs(max(lo, 0) - slo) < 1e-9 and abs(min(hi, 1) - shi) < 1e-9, (name, m, lo, hi, slo, shi)

    # --- Wilson pieces for the 49/50 worked example ---------------------------
    s, n = 49, 50
    p = s / n
    num["z"] = Z
    num["z2"] = Z * Z
    num["w_centre_49of50"] = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    num["w_half_49of50"] = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    num["wald_se_49of50"] = math.sqrt(p * (1 - p) / n)
    num["wilson_lower_at_all_pass_n50"] = n / (n + Z * Z)
    num["cp_lower_at_all_pass_n50"] = 0.025 ** (1 / n)
    num["cp_lower_at_all_pass_n12"] = 0.025 ** (1 / 12)
    num["wilson_lower_at_all_pass_n12"] = 12 / (12 + Z * Z)
    num["rule_of_three_n30"] = 3 / 30
    num["ac_added_successes"] = Z * Z / 2
    num["ac_added_trials"] = Z * Z

    # --- one-sided bounds (Section 2.7) ------------------------------------------
    Z1 = norm.ppf(0.95)
    num["z_one_sided"] = Z1
    num["z1_sq"] = Z1 * Z1
    num["wilson_lower_one_sided_49of50"] = wilson(49, 50, z=Z1)[0]
    num["wilson_lower_one_sided_12of12"] = 12 / (12 + Z1 * Z1)
    num["cp_lower_one_sided_12of12"] = 0.05 ** (1 / 12)
    num["cp_lower_one_sided_49of50"] = beta.ppf(0.05, 49, 2)
    num["jeffreys_lower_one_sided_12of12"] = beta.ppf(0.05, 12.5, 0.5)
    num["n_certify_90_one_sided_wilson"] = math.ceil(9 * Z1 * Z1)
    num["n_certify_90_one_sided_cp"] = math.ceil(math.log(0.05) / math.log(0.9))
    num["n_certify_90_two_sided_wilson"] = math.ceil(9 * Z * Z)
    num["n_certify_90_two_sided_cp"] = math.ceil(math.log(0.025) / math.log(0.9))
    num["cp_upper_one_sided_0of30"] = 1 - 0.05 ** (1 / 30)
    num["wilson_upper_one_sided_0of30"] = Z1 * Z1 / (30 + Z1 * Z1)
    # Jeffreys vs Wilson coverage near the boundary, n = 50
    num["jeffreys_cov_n50_p0.98"] = coverage(jeffreys, 50, 0.98)
    num["jeffreys_cov_n50_p0.95"] = coverage(jeffreys, 50, 0.95)
    num["wilson_cov_n50_p0.95"] = coverage(wilson, 50, 0.95)
    num["jeffreys_cov_n50_p0.5"] = coverage(jeffreys, 50, 0.5)
    inner2 = np.linspace(0.10, 0.90, 161)
    num["jeffreys_min_cov_n50_inner10"] = float(min(coverage(jeffreys, 50, float(pp)) for pp in inner2))
    num["wilson_min_cov_n50_inner10"] = float(min(coverage(wilson, 50, float(pp)) for pp in inner2))

    # --- exact coverage curves at n = 50 ---------------------------------------
    grid = np.linspace(0.01, 0.99, 491)
    curves = {}
    for m in ("wald", "wilson", "clopper_pearson", "jeffreys"):
        curves[m] = [coverage(METHODS[m], 50, float(pp)) for pp in grid]
    num["coverage_grid"] = grid.tolist()
    num["coverage_curves_n50"] = curves

    # --- coverage summary table: min and mean over p in [0.05, 0.95] -----------
    inner = np.linspace(0.05, 0.95, 181)
    table = {}
    for n_ in (20, 50, 164, 1000):
        for m in ("wald", "wilson", "clopper_pearson", "agresti_coull", "jeffreys"):
            c = np.array([coverage(METHODS[m], n_, float(pp)) for pp in inner])
            table[f"{m}_n{n_}_min"] = float(c.min())
            table[f"{m}_n{n_}_mean"] = float(c.mean())
    num["coverage_table"] = table
    # Wald coverage at specific points quoted in the text
    num["wald_cov_n50_p0.98"] = coverage(wald, 50, 0.98)
    num["wald_cov_n50_p0.5"] = coverage(wald, 50, 0.5)
    num["wilson_cov_n50_p0.98"] = coverage(wilson, 50, 0.98)
    num["cp_cov_n50_p0.5"] = coverage(clopper_pearson, 50, 0.5)

    # --- sample-size design (Section 2.8) ---------------------------------------
    for pp, w in ((0.8, 0.02), (0.5, 0.02), (0.9, 0.01), (0.5, 0.01)):
        num[f"n_for_halfwidth_{w}_p{pp}"] = math.ceil(Z * Z * pp * (1 - pp) / (w * w))
    # Wilson-based check: half-width of the Wilson interval at the Wald-designed n
    n_des = num["n_for_halfwidth_0.02_p0.8"]
    lo, hi = wilson(round(0.8 * n_des), n_des)
    num["wilson_halfwidth_at_designed_n"] = (hi - lo) / 2

    # --- 117/164: all five intervals within how much of each other ------------
    los = [num[f"117of164_{m}_lo"] for m in METHODS]
    his = [num[f"117of164_{m}_hi"] for m in METHODS]
    num["117of164_spread_lo_points"] = 100 * (max(los) - min(los))
    num["117of164_spread_hi_points"] = 100 * (max(his) - min(his))

    OUT.write_text(json.dumps(num, indent=2))
    for k, v in num.items():
        if isinstance(v, (int, float)):
            print(f"{k:36s} {v:.5f}")
    print("\ncoverage table (min / mean over p in [0.05, 0.95])")
    for n_ in (20, 50, 164, 1000):
        row = "  ".join(f"{m[:7]:7s} {table[f'{m}_n{n_}_min']:.3f}/{table[f'{m}_n{n_}_mean']:.3f}"
                        for m in ("wald", "wilson", "clopper_pearson", "agresti_coull", "jeffreys"))
        print(f"n={n_:5d}  {row}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
