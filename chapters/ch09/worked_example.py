"""Chapter 9 -- every number printed in the chapter, regenerated from data.

Run:  python chapters/ch09/worked_example.py
Writes chapters/ch09/numbers.json.

Data: data/humaneval_k20.csv (models A and B, 164 items x 20 samples) and
data/humaneval_k20_truth.csv (the latent p_i, used only to check extrapolations).
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm, beta, binom
from scipy.special import betaln, comb
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).with_name("numbers.json")
Z = norm.ppf(0.975)
KS = [1, 2, 5, 10, 20]


# ---------------------------------------------------------------------------
# Estimators (Sections 9.2-9.3)
# ---------------------------------------------------------------------------
def pass_at_k(n, c, k):
    """Unbiased pass@k for one item: 1 - C(n-c, k) / C(n, k), in the stable product form."""
    n, c = int(n), int(c)
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def pass_at_k_naive(n, c, k):
    return 1.0 - (1.0 - c / n) ** k


def pass_pow_k(n, c, k):
    """Unbiased pass^k (all k of k pass) for one item: C(c, k) / C(n, k)."""
    n, c = int(n), int(c)
    if c < k:
        return 0.0
    return float(comb(c, k) / comb(n, k))


def item_bootstrap(values, reps=4000, seed=9):
    """Percentile interval and SE for the mean of per-item values."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    m = values.size
    means = np.array([values[rng.integers(0, m, m)].mean() for _ in range(reps)])
    return float(means.std(ddof=1)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# ---------------------------------------------------------------------------
# Beta model for the p_i (Section 9.4)
# ---------------------------------------------------------------------------
def beta_pass_at_k(a, b, k):
    """1 - E[(1-p)^k] under p ~ Beta(a, b) = 1 - B(a, b+k)/B(a, b)."""
    return 1.0 - math.exp(betaln(a, b + k) - betaln(a, b))


def beta_pass_pow_k(a, b, k):
    return math.exp(betaln(a + k, b) - betaln(a, b))


def beta_fit_counts(c, n):
    """Maximum-likelihood Beta-binomial fit to per-item pass counts."""
    c = np.asarray(c, float)

    def nll(x):
        a, b = np.exp(x)
        return -np.sum(betaln(c + a, n - c + b) - betaln(a, b))

    res = minimize(nll, x0=[0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10})
    return tuple(np.exp(res.x))


def beta_fit_moments(c, n):
    """Method of moments corrected for binomial sampling noise (Chapter 5's decomposition)."""
    ph = np.asarray(c, float) / n
    mean = ph.mean()
    var_between = ph.var(ddof=1) - np.mean(ph * (1 - ph)) / (n - 1)
    var_between = max(var_between, 1e-6)
    common = mean * (1 - mean) / var_between - 1
    return mean * common, (1 - mean) * common


def beta_majority(a, b, k, nodes=4001):
    """P(more than half of k samples pass) under p ~ Beta(a, b), by quadrature."""
    p = np.linspace(0, 1, nodes)[1:-1]
    w = beta.pdf(p, a, b)
    maj = binom.sf(k // 2, k, p) if k % 2 == 1 else binom.sf(k // 2, k, p) + 0.5 * binom.pmf(k // 2, k, p)
    return float(np.trapezoid(w * maj, p))


# ---------------------------------------------------------------------------
# Best-of-n with a verifier (Section 9.5)
# ---------------------------------------------------------------------------
def best_of_n_verifier(p, n, s, t, reps=20000, seed=95):
    """Monte Carlo: n samples, verifier flags each; choose a flagged sample at random,
    else a random sample. Returns P(chosen sample passes)."""
    rng = np.random.default_rng(seed)
    p = np.asarray(p, float)
    wins = 0
    for _ in range(reps):
        i = rng.integers(0, p.size)
        passes = rng.random(n) < p[i]
        flagged = np.where(passes, rng.random(n) < s, rng.random(n) >= t)
        if flagged.any():
            choice = rng.choice(np.where(flagged)[0])
        else:
            choice = rng.integers(0, n)
        wins += passes[choice]
    return wins / reps


def best_of_n_ceiling(p, s, t):
    """n -> infinity limit: choose uniformly among flagged samples."""
    p = np.asarray(p, float)
    return float(np.mean(p * s / (p * s + (1 - p) * (1 - t))))


def main():
    num = {"z": Z}
    d = pd.read_csv(ROOT / "data" / "humaneval_k20.csv")
    truth = pd.read_csv(ROOT / "data" / "humaneval_k20_truth.csv")
    counts = d.groupby(["model", "item"])["pass"].sum().unstack(0)      # items x models
    n = int(d.groupby(["model", "item"]).size().iloc[0])
    m = counts.shape[0]
    num["n_samples"] = n; num["n_items"] = m

    # --- 9.1 the failure-box arithmetic ----------------------------------------------
    num["fb"] = {"pass_at_5_p05": 1 - 0.5 ** 5, "pass_pow_5_p05": 0.5 ** 5,
                 "pass_at_5_p02": 1 - 0.8 ** 5, "pass_pow_5_p02": 0.2 ** 5}

    # --- 9.2-9.3 estimates for A and B --------------------------------------------------
    for mdl in ("A", "B"):
        c = counts[mdl].to_numpy()
        ph = c / n
        res = {"mean_phat": float(ph.mean()), "sd_phat": float(ph.std(ddof=1)),
               "frac_zero": float((c == 0).mean()), "frac_all": float((c == n).mean()),
               "frac_mid": float(((c > 0) & (c < n)).mean())}
        for k in KS:
            v = np.array([pass_at_k(n, ci, k) for ci in c])
            nv = np.array([pass_at_k_naive(n, ci, k) for ci in c])
            pk = np.array([pass_pow_k(n, ci, k) for ci in c])
            se_formula = float(v.std(ddof=1) / math.sqrt(m))
            bse, blo, bhi = item_bootstrap(v)
            res[f"pass_at_{k}"] = float(v.mean()); res[f"se_{k}"] = se_formula
            res[f"boot_se_{k}"] = bse; res[f"lo_{k}"] = blo; res[f"hi_{k}"] = bhi
            res[f"naive_{k}"] = float(nv.mean()); res[f"pass_pow_{k}"] = float(pk.mean())
            res[f"jensen_{k}"] = 1 - (1 - ph.mean()) ** k       # 1 - (1 - p_bar)^k
            # truth
            pt = truth[f"p_{mdl}"].to_numpy()
            res[f"true_pass_at_{k}"] = float((1 - (1 - pt) ** k).mean())
            res[f"true_pass_pow_{k}"] = float((pt ** k).mean())
        # single-sample estimate of pass@1 (the first sample only), for the record
        first = d[(d.model == mdl) & (d["sample"] == 0)]["pass"].mean()
        res["first_sample_pass1"] = float(first)
        res["first_sample_se"] = float(math.sqrt(first * (1 - first) / m))
        # "sample exactly k and check any" for k = 5: use samples 0..4 only
        wide = d[d.model == mdl].pivot(index="item", columns="sample", values="pass").to_numpy()
        any5 = wide[:, :5].max(axis=1)
        res["exact5_pass_at_5"] = float(any5.mean()); res["exact5_se"] = float(any5.std(ddof=1) / math.sqrt(m))
        # Beta fits
        a_ml, b_ml = beta_fit_counts(c, n); a_mo, b_mo = beta_fit_moments(c, n)
        res["beta_ml"] = [float(a_ml), float(b_ml)]; res["beta_mo"] = [float(a_mo), float(b_mo)]
        res["beta_mean"] = float(a_ml / (a_ml + b_ml))
        pt = truth[f"p_{mdl}"].to_numpy()
        res["beta_pred"] = {}; res["beta_true"] = {}
        for k in (1, 5, 10, 20, 50, 100, 200):
            res["beta_pred"][str(k)] = beta_pass_at_k(a_ml, b_ml, k)
            res["beta_true"][str(k)] = float((1 - (1 - pt) ** k).mean())
        res["beta_pass_pow"] = {str(k): beta_pass_pow_k(a_ml, b_ml, k) for k in (1, 2, 5, 10)}
        # extrapolation from the naive plug-in (no Beta): 1 - (1 - phat_i)^k averaged
        res["plugin_pred"] = {str(k): float((1 - (1 - ph) ** k).mean()) for k in (50, 100, 200)}
        # majority vote under the fitted Beta
        res["majority"] = {str(k): beta_majority(a_ml, b_ml, k) for k in (1, 3, 5, 11, 21)}
        # true majority (from latent p)
        res["majority_true"] = {str(k): float(np.mean(binom.sf(k // 2, k, pt))) for k in (3, 5, 11, 21)}
        num[mdl] = res

    # --- paired difference A - B at each k (Chapter 3's paired SE over items) ---------------
    num["diff"] = {}
    for k in KS:
        va = np.array([pass_at_k(n, ci, k) for ci in counts["A"]])
        vb = np.array([pass_at_k(n, ci, k) for ci in counts["B"]])
        w = va - vb
        num["diff"][str(k)] = {"diff": float(w.mean()), "se": float(w.std(ddof=1) / math.sqrt(m)),
                               "se_unpaired": float(math.sqrt(va.var(ddof=1) / m + vb.var(ddof=1) / m))}

    # --- 9.5 best-of-n with a verifier (judges J and L of Chapter 8), model A -----------------
    pt = truth["p_A"].to_numpy()
    num["verifier"] = {}
    for jn, (s, t) in {"J": (0.90, 0.70), "L": (0.96, 0.90), "perfect": (1.0, 1.0)}.items():
        row = {"s": s, "t": t, "ceiling": best_of_n_ceiling(pt, s, t)}
        for nn in (1, 5, 20):
            row[f"n{nn}"] = best_of_n_verifier(pt, nn, s, t)
        num["verifier"][jn] = row
    num["verifier"]["pass_at_5_true"] = num["A"]["true_pass_at_5"]
    num["verifier"]["pass_at_20_true"] = num["A"]["true_pass_at_20"]

    # --- 9.6 cost: illustrative cost per sample, A = 1.6 x B --------------------------------
    cost = {"A": 1.6, "B": 1.0}
    num["cost"] = {"per_sample": cost, "table": {}}
    for budget in (2, 4, 8, 16):
        row = {}
        for mdl in ("A", "B"):
            k = int(budget // cost[mdl])
            if k >= 1:
                v = np.array([pass_at_k(n, ci, min(k, n)) for ci in counts[mdl]])
                row[mdl] = {"k": min(k, n), "pass_at_k": float(v.mean())}
        num["cost"]["table"][str(budget)] = row
    # expected cost per solved problem at k = 1: cost / pass@1
    for mdl in ("A", "B"):
        num["cost"][f"cost_per_solve_{mdl}"] = cost[mdl] / num[mdl]["pass_at_1"]

    # --- 9.3 variance of the per-item estimator, checked by simulation at p = 0.3 ----------
    rng = np.random.default_rng(93)
    p0 = 0.3; reps = 20000
    for k in (1, 5, 10):
        sims = np.array([pass_at_k(n, rng.binomial(n, p0), k) for _ in range(reps)])
        num[f"sim_p03_k{k}"] = {"mean": float(sims.mean()), "true": 1 - (1 - p0) ** k, "sd": float(sims.std(ddof=1)),
                                "naive_mean": float(np.mean([pass_at_k_naive(n, rng.binomial(n, p0), k) for _ in range(reps)]))}

    OUT.write_text(json.dumps(num, indent=2))
    for k, v in num.items():
        print(k, v if not isinstance(v, dict) else json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}, default=str)[:1500])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
