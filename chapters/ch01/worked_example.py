"""Chapter 1 -- every number printed in the chapter, regenerated from data.

Run:  python chapters/ch01/worked_example.py
Writes chapters/ch01/numbers.json and prints a human-readable summary.
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "humaneval_k20.csv"
OUT = pathlib.Path(__file__).with_name("numbers.json")

Z = 1.959963984540054  # 97.5th percentile of the standard normal


def se_binary(results):
    """Plug-in standard error of a proportion from a 0/1 array (Section 1.4)."""
    x = np.asarray(results, dtype=float)
    n = x.size
    p_hat = x.mean()
    return math.sqrt(p_hat * (1 - p_hat) / n)


def main():
    df = pd.read_csv(DATA)
    first = df[df["sample"] == 0]
    a = first[first.model == "A"].sort_values("item")["pass"].to_numpy()
    b = first[first.model == "B"].sort_values("item")["pass"].to_numpy()
    n = a.size
    num = {}

    # --- 1.1 The failure -------------------------------------------------
    num["n"] = int(n)
    num["correct_A"], num["correct_B"] = int(a.sum()), int(b.sum())
    num["p_A"], num["p_B"] = a.mean(), b.mean()
    num["gap_points"] = 100 * (a.mean() - b.mean())
    num["se_A"], num["se_B"] = se_binary(a), se_binary(b)
    num["se_A_points"] = 100 * num["se_A"]
    num["ci_A_lo"], num["ci_A_hi"] = num["p_A"] - Z * num["se_A"], num["p_A"] + Z * num["se_A"]
    num["ci_B_lo"], num["ci_B_hi"] = num["p_B"] - Z * num["se_B"], num["p_B"] + Z * num["se_B"]

    # --- 1.4 sqrt(n) law: items needed for a given SE at p = p_A ----------
    p = num["p_A"]
    for target_pts in (2, 1, 0.5):
        num[f"n_for_se_{target_pts}pt"] = math.ceil(p * (1 - p) / (target_pts / 100) ** 2)
    num["se_p_half_n164_points"] = 100 * math.sqrt(0.25 / n)  # worst case
    num["var_bernoulli_at_pA"] = p * (1 - p)

    # --- 1.5 finite population correction --------------------------------
    for N in (1_000, 10_000, 1_000_000):
        num[f"fpc_N{N}"] = math.sqrt((N - n) / (N - 1))

    # --- 1.6 worked example: benchmark resolution table ------------------
    # Public test-set sizes (fixed constants, cited in the chapter).
    sizes = {
        "HumanEval": 164, "GPQA Diamond": 198, "MBPP (test)": 500,
        "SWE-bench Verified": 500, "IFEval": 541, "ARC-Challenge (test)": 1172,
        "GSM8K (test)": 1319, "MATH (test)": 5000, "BBH": 6511,
        "HellaSwag (val)": 10042, "MMLU (test)": 14042,
    }
    table = []
    for name, size in sizes.items():
        for pp in (0.5, 0.7, 0.9):
            table.append({"benchmark": name, "n": size, "p": pp,
                          "se_points": 100 * math.sqrt(pp * (1 - pp) / size)})
    num["resolution_table"] = table
    num["se_mmlu_pA_points"] = 100 * math.sqrt(p * (1 - p) / 14042)

    # --- 1.7 two identical models on independent 164-item draws -----------
    # analytic: diff ~ N(0, 2 p(1-p)/n); P(|diff| > 1 point)
    sd_diff = math.sqrt(2 * p * (1 - p) / n)
    num["sd_diff_indep_points"] = 100 * sd_diff
    num["p_gap_gt_1pt_analytic"] = 2 * (1 - norm.cdf(0.01 / sd_diff))
    num["p_gap_gt_gap_analytic"] = 2 * (1 - norm.cdf((num["gap_points"] / 100) / sd_diff))
    # simulation with the same p for both models
    rng = np.random.default_rng(1)
    reps = 200_000
    xa = rng.binomial(n, p, reps) / n
    xb = rng.binomial(n, p, reps) / n
    num["sim_reps"] = reps
    num["p_gap_gt_1pt_sim"] = float(np.mean(np.abs(xa - xb) > 0.01 - 1e-12))
    num["p_gap_ge_observed_sim"] = float(np.mean(np.abs(xa - xb) >= (2 / n) - 1e-12))
    # the observed gap is exactly 2 items; what is P(|diff| >= 2 items)?
    num["gap_items"] = int(a.sum() - b.sum())

    # --- 1.8 the same benchmark run twice on the same model ---------------
    # Model A, sample 0 vs sample 1: the score changes with nothing but the seed
    a1 = df[(df["sample"] == 1) & (df.model == "A")].sort_values("item")["pass"].to_numpy()
    num["correct_A_run2"] = int(a1.sum())
    num["p_A_run2"] = a1.mean()
    runs = df[df.model == "A"].groupby("sample")["pass"].mean().to_numpy()
    num["A_run_scores_min"], num["A_run_scores_max"] = float(runs.min()), float(runs.max())
    num["A_run_scores_sd_points"] = 100 * float(runs.std(ddof=1))

    # --- checks --------------------------------------------------------------
    assert abs(num["se_A"] - math.sqrt(p * (1 - p) / n)) < 1e-12

    OUT.write_text(json.dumps(num, indent=2))
    for k, v in num.items():
        if k != "resolution_table":
            print(f"{k:32s} {v}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
