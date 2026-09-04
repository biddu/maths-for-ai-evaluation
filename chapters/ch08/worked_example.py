"""Chapter 8 -- every number printed in the chapter, regenerated from data.

Run:  python chapters/ch08/worked_example.py
Writes chapters/ch08/numbers.json.

Data: data/judge2000.csv (model K's responses judged by J and L, 200-item human
calibration subset) and data/judge_pairs.csv (judge J on E-versus-F pairs in
both orders).
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm, beta, binom

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).with_name("numbers.json")
Z = norm.ppf(0.975)


# ---------------------------------------------------------------------------
# The judge as a diagnostic test (Sections 8.1-8.3)
# ---------------------------------------------------------------------------
def observed_rate(theta, s, t):
    """Mixture identity: judge-positive rate for true rate theta."""
    return s * theta + (1 - t) * (1 - theta)


def rogan_gladen(q, s, t, clip=True):
    th = (q + t - 1) / (s + t - 1)
    return min(1.0, max(0.0, th)) if clip else th


def rg_variance(q, N, s, n1, t, n0):
    """Delta-method variance of the Rogan-Gladen estimate with q, s, t estimated
    from N judged items, n1 calibration positives and n0 calibration negatives."""
    J = s + t - 1
    th = rogan_gladen(q, s, t, clip=False)
    return (q * (1 - q) / N + th ** 2 * s * (1 - s) / n1 + (1 - th) ** 2 * t * (1 - t) / n0) / J ** 2


def rg_floor(theta, s, t, n_h):
    """Variance of the corrected estimate as N -> infinity, calibration of size n_h
    drawn from the same population (n1 = theta n_h, n0 = (1 - theta) n_h)."""
    J = s + t - 1
    return (theta * s * (1 - s) + (1 - theta) * t * (1 - t)) / (n_h * J ** 2)


def judge_worth_it(theta, s, t):
    """True when the corrected estimate beats human-only labels at N -> infinity."""
    J = s + t - 1
    return theta * s * (1 - s) + (1 - theta) * t * (1 - t) < J ** 2 * theta * (1 - theta)


def budget_split(theta, s, t, cost_h, cost_j):
    """Optimal ratio n_h / N of calibration to judged items for a fixed budget."""
    J = s + t - 1
    q = observed_rate(theta, s, t)
    A = q * (1 - q) / J ** 2
    B = (theta * s * (1 - s) + (1 - theta) * t * (1 - t)) / J ** 2
    return math.sqrt(B * cost_j / (A * cost_h))


def calibration_size(theta, s, t, h, z=Z):
    """Calibration items needed for a half-width h at N -> infinity (random calibration sample)."""
    J = s + t - 1
    return z ** 2 * (theta * s * (1 - s) + (1 - theta) * t * (1 - t)) / (J ** 2 * h ** 2)


def multiclass_correct(q_hat, M_hat):
    """theta_hat = (M^T)^{-1} q_hat for a K x K confusion matrix M[j, k] = P(F = k | Y = j)."""
    return np.linalg.solve(M_hat.T, q_hat)


def multiclass_var_q(q_hat, M_hat, N):
    """Delta-method covariance from the judged items alone (M treated as known)."""
    Sigma_q = (np.diag(q_hat) - np.outer(q_hat, q_hat)) / N
    Minv = np.linalg.inv(M_hat.T)
    return Minv @ Sigma_q @ Minv.T


# ---------------------------------------------------------------------------
# Prediction-powered inference (Section 8.5)
# ---------------------------------------------------------------------------
def ppi_mean(f_unlab, f_lab, y_lab, lam=1.0):
    """PPI estimate of E[y] from judge labels f on N unlabelled items and
    (f, y) pairs on n labelled items; lam = 1 is plain PPI."""
    f_unlab, f_lab, y_lab = (np.asarray(a, float) for a in (f_unlab, f_lab, y_lab))
    N, n = f_unlab.size, f_lab.size
    est = lam * f_unlab.mean() + (y_lab - lam * f_lab).mean()
    var = lam ** 2 * f_unlab.var(ddof=1) / N + (y_lab - lam * f_lab).var(ddof=1) / n
    return est, math.sqrt(var)


def ppi_lambda(f_unlab, f_lab, y_lab):
    """Variance-minimising lambda (PPI++)."""
    N, n = len(f_unlab), len(f_lab)
    f_lab, y_lab = np.asarray(f_lab, float), np.asarray(y_lab, float)
    cov = np.cov(f_lab, y_lab, ddof=1)[0, 1]
    vf = np.var(np.concatenate([f_unlab, f_lab]), ddof=1)
    return cov / vf * N / (N + n)


# ---------------------------------------------------------------------------
# Bayesian version (Section 8.6): Beta priors, posterior for theta on a grid,
# integrating s and t over their Beta posteriors on a grid.
# ---------------------------------------------------------------------------
def bayes_posterior(yJ, N, a1, n1, b0, n0, h1, h0, grid=2001, sgrid=241, prior=(1, 1)):
    """yJ judge positives among N judged-only items; a1 of n1 calibration positives
    judged positive; b0 of n0 calibration negatives judged negative; h1/h0 human
    positives/negatives in the calibration set. Returns (theta grid, posterior density)."""
    th = np.linspace(0, 1, grid)
    s_nodes = np.linspace(0.0005, 0.9995, sgrid)
    t_nodes = np.linspace(0.0005, 0.9995, sgrid)
    ws = beta.pdf(s_nodes, 1 + a1, 1 + n1 - a1); ws /= ws.sum()
    wt = beta.pdf(t_nodes, 1 + b0, 1 + n0 - b0); wt /= wt.sum()
    # likelihood of the judge count given theta, averaged over (s, t)
    like = np.zeros(grid)
    for si, s in enumerate(s_nodes):
        if ws[si] < 1e-12:
            continue
        q = s * th[:, None] + (1 - t_nodes[None, :]) * (1 - th[:, None])       # grid x tgrid
        like += ws[si] * (binom.pmf(yJ, N, np.clip(q, 1e-12, 1 - 1e-12)) * wt[None, :]).sum(1)
    post = beta.pdf(th, prior[0] + h1, prior[1] + h0) * like
    post /= np.trapezoid(post, th)
    return th, post


def posterior_summary(th, post):
    cdf = np.cumsum(post) * (th[1] - th[0])
    mean = np.trapezoid(th * post, th)
    lo, hi = np.interp([0.025, 0.975], cdf, th)
    return mean, lo, hi


def metropolis(yJ, N, a1, n1, b0, n0, h1, h0, draws=60000, seed=8):
    """Cross-check: random-walk Metropolis on (theta, s, t) with flat priors."""
    rng = np.random.default_rng(seed)

    def logpost(x):
        th, s, t = x
        if not (0 < th < 1 and 0 < s < 1 and 0 < t < 1):
            return -np.inf
        q = observed_rate(th, s, t)
        return (yJ * math.log(q) + (N - yJ) * math.log(1 - q) + a1 * math.log(s) + (n1 - a1) * math.log(1 - s)
                + b0 * math.log(t) + (n0 - b0) * math.log(1 - t) + h1 * math.log(th) + h0 * math.log(1 - th))

    x = np.array([0.5, 0.8, 0.7]); lp = logpost(x); out = np.empty((draws, 3))
    for i in range(draws):
        y = x + rng.normal(0, [0.05, 0.03, 0.04])
        lpy = logpost(y)
        if math.log(rng.random()) < lpy - lp:
            x, lp = y, lpy
        out[i] = x
    return out[10000:]


# ---------------------------------------------------------------------------
# Position bias (Section 8.8): McNemar on the swapped orders
# ---------------------------------------------------------------------------
def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * binom.cdf(k, n, 0.5))


def main():
    num = {"z": Z}
    # --- 8.1 the failure-box arithmetic ---------------------------------------
    num["fb_q"] = observed_rate(0.5, 0.9, 0.7)
    num["fb_fp"] = 0.5 * 0.3; num["fb_fn"] = 0.5 * 0.1
    num["fb_rg"] = rogan_gladen(0.6, 0.9, 0.7)

    d = pd.read_csv(ROOT / "data" / "judge2000.csv")
    cal = d[d.calib == 1]; rest = d[d.calib == 0]
    N, n = len(rest), len(cal)
    num["N_judged_only"] = int(N); num["n_calib"] = int(n)
    num["truth_rate_all"] = float(d.truth.mean())
    num["truth_rate_rest"] = float(rest.truth.mean())
    h1 = int(cal.human.sum()); h0 = n - h1
    num["h1"] = h1; num["h0"] = h0
    num["human_only"] = {"est": h1 / n, "se": math.sqrt(h1 / n * (1 - h1 / n) / n)}
    num["human_only"]["lo"] = num["human_only"]["est"] - Z * num["human_only"]["se"]
    num["human_only"]["hi"] = num["human_only"]["est"] + Z * num["human_only"]["se"]

    rng = np.random.default_rng(88)
    for jname, (s_true, t_true) in {"J": (0.90, 0.70), "L": (0.96, 0.90)}.items():
        f = rest[jname].to_numpy(); fc = cal[jname].to_numpy(); y = cal.human.to_numpy().astype(int)
        yJ = int(f.sum()); q = yJ / N
        a1 = int(fc[y == 1].sum()); n1 = int((y == 1).sum())
        b0 = int((1 - fc[y == 0]).sum()); n0 = int((y == 0).sum())
        s_hat, t_hat = a1 / n1, b0 / n0
        Jy = s_hat + t_hat - 1
        th = rogan_gladen(q, s_hat, t_hat)
        var = rg_variance(q, N, s_hat, n1, t_hat, n0)
        # the three variance pieces, as shares
        pieces = np.array([q * (1 - q) / N, th ** 2 * s_hat * (1 - s_hat) / n1, (1 - th) ** 2 * t_hat * (1 - t_hat) / n0]) / Jy ** 2
        # bootstrap: resample judged items, calibration positives and negatives
        B = 4000; boots = np.empty(B)
        for r in range(B):
            qb = rng.binomial(N, q) / N
            sb = rng.binomial(n1, s_hat) / n1; tb = rng.binomial(n0, t_hat) / n0
            boots[r] = rogan_gladen(qb, sb, tb)
        # PPI
        est_pp, se_pp = ppi_mean(f, fc, y)
        lam = ppi_lambda(f, fc, y)
        est_pl, se_pl = ppi_mean(f, fc, y, lam)
        # Bayes
        thg, post = bayes_posterior(yJ, N, a1, n1, b0, n0, h1, h0)
        pm, plo, phi = posterior_summary(thg, post)
        mc = metropolis(yJ, N, a1, n1, b0, n0, h1, h0)
        num[f"judge_{jname}"] = {
            "s_true": s_true, "t_true": t_true,
            "yJ": yJ, "q": q, "q_se": math.sqrt(q * (1 - q) / N),
            "naive_lo": q - Z * math.sqrt(q * (1 - q) / N), "naive_hi": q + Z * math.sqrt(q * (1 - q) / N),
            "a1": a1, "n1": n1, "b0": b0, "n0": n0, "s_hat": s_hat, "t_hat": t_hat, "youden": Jy,
            "s_se": math.sqrt(s_hat * (1 - s_hat) / n1), "t_se": math.sqrt(t_hat * (1 - t_hat) / n0),
            "rg": th, "rg_se": math.sqrt(var), "rg_lo": th - Z * math.sqrt(var), "rg_hi": th + Z * math.sqrt(var),
            "rg_pieces_share": (pieces / pieces.sum()).tolist(), "rg_pieces_se": np.sqrt(pieces).tolist(),
            "boot_se": float(boots.std(ddof=1)), "boot_lo": float(np.quantile(boots, 0.025)), "boot_hi": float(np.quantile(boots, 0.975)),
            "floor_se": math.sqrt(rg_floor(th, s_hat, t_hat, n)),
            "worth_it": bool(judge_worth_it(0.5, s_true, t_true)),
            "ppi": est_pp, "ppi_se": se_pp, "ppi_lo": est_pp - Z * se_pp, "ppi_hi": est_pp + Z * se_pp,
            "lam": lam, "ppi_pp": est_pl, "ppi_pp_se": se_pl, "ppi_pp_lo": est_pl - Z * se_pl, "ppi_pp_hi": est_pl + Z * se_pl,
            "bayes_mean": pm, "bayes_lo": plo, "bayes_hi": phi,
            "mc_mean": float(mc[:, 0].mean()), "mc_lo": float(np.quantile(mc[:, 0], 0.025)), "mc_hi": float(np.quantile(mc[:, 0], 0.975)),
            "mc_s": float(mc[:, 1].mean()), "mc_t": float(mc[:, 2].mean()),
            "err_rate_calib": float((fc != y).mean()),
            "n_for_rg_beats_human": N * (0.25 - rg_floor(0.5, s_true, t_true, 1)) / (observed_rate(0.5, s_true, t_true) * (1 - observed_rate(0.5, s_true, t_true)) / (s_true + t_true - 1) ** 2),
            "budget_ratio_cost10": budget_split(0.5, s_true, t_true, 10, 1),
            "budget_ratio_cost50": budget_split(0.5, s_true, t_true, 50, 1),
        }
        assert abs(pm - num[f"judge_{jname}"]["mc_mean"]) < 0.01, (pm, num[f"judge_{jname}"]["mc_mean"])

    # --- 8.4 regime map: at N -> infinity, which (s, t) beat human-only? ------------
    num["regime"] = {}
    for theta in (0.2, 0.5, 0.8):
        # smallest symmetric accuracy s = t at which the judge helps
        for a in np.linspace(0.5, 1, 5001):
            if judge_worth_it(theta, a, a):
                num["regime"][f"symmetric_threshold_theta{theta}"] = float(a); break
    num["regime"]["J_ratio_floor_to_human"] = math.sqrt(rg_floor(0.5, 0.9, 0.7, 200) / (0.25 / 200))
    num["regime"]["L_ratio_floor_to_human"] = math.sqrt(rg_floor(0.5, 0.96, 0.90, 200) / (0.25 / 200))

    # simulation check of the delta-method interval (judge J, true values)
    reps = 4000; cover = 0; ses = []; ests = []
    rng = np.random.default_rng(89)
    for r in range(reps):
        truth = rng.random(N + n) < 0.5
        u = rng.random(N + n)
        f = np.where(truth, u < 0.9, u >= 0.7)
        y = truth[:n]; fc = f[:n]; fr = f[n:]
        q = fr.mean(); n1 = y.sum(); n0 = n - n1
        s_hat = fc[y].mean(); t_hat = 1 - fc[~y].mean()
        th = rogan_gladen(q, s_hat, t_hat); se = math.sqrt(rg_variance(q, N, s_hat, n1, t_hat, n0))
        cover += (th - Z * se <= 0.5 <= th + Z * se); ses.append(se); ests.append(th)
    num["sim_J"] = {"coverage": cover / reps, "mean_se": float(np.mean(ses)), "sd_est": float(np.std(ests, ddof=1)), "reps": reps}

    # --- 8.8 position bias -----------------------------------------------------------
    p = pd.read_csv(ROOT / "data" / "judge_pairs.csv")
    EF, FE, REP = p.EF.to_numpy(), p.FE.to_numpy(), p.EF_rep.to_numpy()
    M = len(p)
    num["pairs"] = {"n": int(M),
                    "p_E_first": float(EF.mean()), "p_E_second": float(FE.mean()),
                    "repeat_consistency": float((EF == REP).mean()), "order_consistency": float((EF == FE).mean()),
                    "first_wins": float(((EF == 1).sum() + (FE == 0).sum()) / (2 * M))}
    # McNemar on the discordant pairs: b = E wins only when first, c = E wins only when second
    b = int(((EF == 1) & (FE == 0)).sum()); c = int(((EF == 0) & (FE == 1)).sum())
    num["pairs"].update({"b_first_only": b, "c_second_only": c, "mcnemar_p": mcnemar_exact(b, c),
                         "mcnemar_chi2": (b - c) ** 2 / (b + c), "bias_est": (b - c) / M,
                         "bias_se": math.sqrt((b + c) / M - ((b - c) / M) ** 2) / math.sqrt(M)})
    # repeat: the same test on EF vs EF_rep should NOT reject
    b2 = int(((EF == 1) & (REP == 0)).sum()); c2 = int(((EF == 0) & (REP == 1)).sum())
    num["pairs"].update({"rep_b": b2, "rep_c": c2, "rep_mcnemar_p": mcnemar_exact(b2, c2)})
    # position-free verdict: E wins if it wins in both orders; tie if split
    both = ((EF == 1) & (FE == 1)).mean(); neither = ((EF == 0) & (FE == 0)).mean()
    num["pairs"].update({"E_both": float(both), "F_both": float(neither), "split": float(1 - both - neither)})
    # agreement with the human on the labelled subset, by order and debiased
    hl = p.dropna(subset=["human"]); hy = hl.human.astype(int).to_numpy()
    num["pairs"].update({"n_human": int(len(hl)),
                         "agree_EF": float((hl.EF == hy).mean()), "agree_FE": float((hl.FE == hy).mean()),
                         "agree_both_orders": float(((hl.EF == hy) & (hl.FE == hy)).mean())})
    # verbosity: among human-labelled pairs, judge (E-first order) agreement with the human
    # when the human-preferred response is the longer one vs the shorter one
    longer_is_E = (hl.len_E > hl.len_F).to_numpy()
    human_prefers_longer = (hy == 1) == longer_is_E
    agree = (hl.EF.to_numpy() == hy)
    a_long = agree[human_prefers_longer].mean(); a_short = agree[~human_prefers_longer].mean()
    n_long = int(human_prefers_longer.sum()); n_short = int((~human_prefers_longer).sum())
    se_diff = math.sqrt(a_long * (1 - a_long) / n_long + a_short * (1 - a_short) / n_short)
    num["verbosity"] = {"agree_when_human_prefers_longer": float(a_long), "n_long": n_long,
                        "agree_when_human_prefers_shorter": float(a_short), "n_short": n_short,
                        "diff": float(a_long - a_short), "se": se_diff, "z": float((a_long - a_short) / se_diff),
                        "p": float(2 * norm.sf(abs(a_long - a_short) / se_diff))}
    # judge's overall rate of preferring the longer response vs the human's
    num["verbosity"]["judge_prefers_longer"] = float(((p.EF == 1) == (p.len_E > p.len_F)).mean())
    num["verbosity"]["human_prefers_longer"] = float(human_prefers_longer.mean())

    # --- 8.4 calibration size for a target half-width ------------------------------
    num["calib_size"] = {f"{jn}_h{h}": calibration_size(0.5, st[0], st[1], h)
                         for jn, st in {"J": (0.9, 0.7), "L": (0.96, 0.9)}.items() for h in (0.05, 0.03)}

    # --- 8.6 more than two labels: a simulated three-class judge -----------------------
    rng = np.random.default_rng(90)
    theta3 = np.array([0.5, 0.3, 0.2])                       # good / partial / bad
    M3 = np.array([[0.85, 0.12, 0.03],                       # rows: true class; cols: judge class
                   [0.20, 0.65, 0.15],
                   [0.05, 0.25, 0.70]])
    N3, n3 = 2000, 300
    truth = rng.choice(3, N3 + n3, p=theta3)
    judged = np.array([rng.choice(3, p=M3[y]) for y in truth])
    y_cal, f_cal = truth[:n3], judged[:n3]; f_rest = judged[n3:]
    q_hat = np.bincount(f_rest, minlength=3) / N3
    M_hat = np.array([np.bincount(f_cal[y_cal == j], minlength=3) / max(1, (y_cal == j).sum()) for j in range(3)])
    th3 = multiclass_correct(q_hat, M_hat)
    V_q = multiclass_var_q(q_hat, M_hat, N3)
    # parametric bootstrap over both the judged counts and the calibration rows
    B = 4000; boots = np.empty((B, 3))
    for r in range(B):
        qb = rng.multinomial(N3, q_hat) / N3
        Mb = np.array([rng.multinomial(int((y_cal == j).sum()), M_hat[j]) / (y_cal == j).sum() for j in range(3)])
        boots[r] = multiclass_correct(qb, Mb)
    num["multiclass"] = {
        "theta_true": theta3.tolist(), "M": M3.tolist(), "N": N3, "n_calib": n3,
        "truth_rate_rest": (np.bincount(truth[n3:], minlength=3) / N3).tolist(),
        "n_cal_by_class": np.bincount(y_cal, minlength=3).tolist(),
        "q_hat": q_hat.tolist(), "M_hat": M_hat.tolist(), "theta_hat": th3.tolist(),
        "se_q_only": np.sqrt(np.diag(V_q)).tolist(), "se_boot": boots.std(axis=0, ddof=1).tolist(),
        "cond_M": float(np.linalg.cond(M3.T)), "cond_binary_J": float(np.linalg.cond(np.array([[0.7, 0.3], [0.1, 0.9]]))),
        "cond_binary_L": float(np.linalg.cond(np.array([[0.9, 0.1], [0.04, 0.96]]))),
        "det_M": float(np.linalg.det(M3)),
        "naive_se": np.sqrt(q_hat * (1 - q_hat) / N3).tolist(),
    }

    OUT.write_text(json.dumps(num, indent=2))
    for k, v in num.items():
        print(k, v if not isinstance(v, dict) else {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
