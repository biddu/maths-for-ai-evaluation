"""Chapter 11 -- every number printed in the chapter, regenerated from data.

Run:  python chapters/ch11/worked_example.py
Writes chapters/ch11/numbers.json.

Data: data/arena300.csv (300 battles among M1..M6 judged by J, with ties and a
first-position bias). The true strengths are stated in generate/make_arena300.py.
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).with_name("numbers.json")
Z = norm.ppf(0.975)
MODELS = [f"M{i}" for i in range(1, 7)]
K = len(MODELS)
TRUE_BETA = np.array([1.00, 0.45, 0.40, 0.35, 0.30, -0.60]); TRUE_BETA -= TRUE_BETA.mean()


# ---------------------------------------------------------------------------
# Data -> win matrix (ties counted as half a win each, the arena convention)
# ---------------------------------------------------------------------------
def win_matrix(df, tie_weight=0.5):
    """W[i, j] = (fractional) number of wins of i over j."""
    W = np.zeros((K, K))
    idx = {m: k for k, m in enumerate(MODELS)}
    for _, r in df.iterrows():
        i, j = idx[r["first"]], idx[r["second"]]
        if r["outcome"] == "first":
            W[i, j] += 1
        elif r["outcome"] == "second":
            W[j, i] += 1
        else:
            W[i, j] += tie_weight; W[j, i] += tie_weight
    return W


# ---------------------------------------------------------------------------
# Bradley-Terry by the MM algorithm (Hunter 2004), Section 11.2
# ---------------------------------------------------------------------------
def bt_mm(W, iters=2000, tol=1e-12):
    """Returns beta (centred), the number of iterations, and the log-likelihood."""
    N = W + W.T                     # comparisons per pair
    w = W.sum(1)                    # wins per model
    pi = np.ones(K)
    for it in range(iters):
        denom = np.array([sum(N[i, j] / (pi[i] + pi[j]) for j in range(K) if j != i) for i in range(K)])
        new = w / denom
        new /= np.exp(np.mean(np.log(new)))          # geometric-mean normalisation
        if np.max(np.abs(np.log(new) - np.log(pi))) < tol:
            pi = new; break
        pi = new
    beta = np.log(pi); beta -= beta.mean()
    return beta, it + 1, bt_loglik(beta, W)


def bt_loglik(beta, W):
    ll = 0.0
    for i in range(K):
        for j in range(K):
            if i != j and W[i, j] > 0:
                ll += W[i, j] * (beta[i] - np.logaddexp(beta[i], beta[j]))
    return ll


def bt_fisher_cov(beta, W):
    """Asymptotic covariance of the centred estimate: pseudo-inverse of the information."""
    N = W + W.T
    I = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            if N[i, j] > 0:
                p = 1 / (1 + math.exp(beta[j] - beta[i]))
                v = N[i, j] * p * (1 - p)
                I[i, i] += v; I[j, j] += v; I[i, j] -= v; I[j, i] -= v
    return np.linalg.pinv(I)


def se_diff(cov, i, j):
    return math.sqrt(cov[i, i] + cov[j, j] - 2 * cov[i, j])


# ---------------------------------------------------------------------------
# Logistic-regression form with a position term, Section 11.3
# ---------------------------------------------------------------------------
def design(df):
    """Rows: one per battle. X[:, k] = +1 if model k shown first, -1 if second; last column = 1 (position)."""
    idx = {m: k for k, m in enumerate(MODELS)}
    X = np.zeros((len(df), K + 1)); y = np.zeros(len(df))
    for r, (_, row) in enumerate(df.iterrows()):
        X[r, idx[row["first"]]] = 1; X[r, idx[row["second"]]] = -1; X[r, K] = 1
        y[r] = {"first": 1.0, "second": 0.0, "tie": 0.5}[row["outcome"]]
    return X, y


def logistic_fit(X, y, drop_position=False):
    """Maximum likelihood with y in [0, 1] (ties as 0.5); sum-to-zero constraint on betas
    imposed by dropping the last model's column and reconstructing."""
    cols = list(range(K - 1)) + ([] if drop_position else [K])
    Xr = X[:, cols] - (X[:, K - 1][:, None] * np.array([1.0] * (K - 1) + [0.0] * (0 if drop_position else 1)))
    # beta_K = -sum(beta_1..K-1)  =>  contribution x_K beta_K = -x_K sum(beta_k): handled by subtracting x_K from each model column

    def nll(th):
        eta = Xr @ th
        return -np.sum(y * eta - np.logaddexp(0, eta))

    def grad(th):
        eta = Xr @ th
        p = 1 / (1 + np.exp(-eta))
        return -Xr.T @ (y - p)

    res = minimize(nll, np.zeros(Xr.shape[1]), jac=grad, method="BFGS", options={"gtol": 1e-10})
    th = res.x
    beta = np.append(th[:K - 1], -th[:K - 1].sum())
    gamma = None if drop_position else th[K - 1]
    # covariance from the observed information
    eta = Xr @ th; p = 1 / (1 + np.exp(-eta))
    H = Xr.T @ (Xr * (p * (1 - p))[:, None])
    cov_th = np.linalg.inv(H)
    return beta, gamma, cov_th, -res.fun


def elo(df, k_factor=32, order=None, start=1000.0):
    R = {m: start for m in MODELS}
    seq = df if order is None else df.iloc[order]
    for _, r in seq.iterrows():
        a, b = r["first"], r["second"]
        s = {"first": 1.0, "second": 0.0, "tie": 0.5}[r["outcome"]]
        e = 1 / (1 + 10 ** ((R[b] - R[a]) / 400))
        R[a] += k_factor * (s - e); R[b] -= k_factor * (s - e)
    return R


def bootstrap_bt(df, reps, rng):
    betas = np.empty((reps, K)); ranks = np.empty((reps, K), int)
    n = len(df)
    for r in range(reps):
        samp = df.iloc[rng.integers(0, n, n)]
        b, _, _ = bt_mm(win_matrix(samp))
        betas[r] = b; ranks[r] = (-b).argsort().argsort() + 1
    return betas, ranks


# ---------------------------------------------------------------------------
# Active pair selection, Section 11.7
# ---------------------------------------------------------------------------
def simulate_battle(i, j, rng, nu=0.25, position=0.0):
    pi_i, pi_j = math.exp(TRUE_BETA[i]), math.exp(TRUE_BETA[j])
    tie = nu * math.sqrt(pi_i * pi_j)
    probs = np.array([pi_i, pi_j, tie]) / (pi_i + pi_j + tie)
    return rng.choice(3, p=probs)      # 0: i wins, 1: j wins, 2: tie


def run_schedule(strategy, n_total, rng, warm=60, every=1):
    """Returns the history of the largest SE among the four contested models' pairwise gaps."""
    W = np.zeros((K, K)); hist = []
    contested = [1, 2, 3, 4]
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    for t in range(n_total):
        if t < warm or strategy == "uniform":
            i, j = pairs[rng.integers(len(pairs))]
        else:
            beta, _, _ = bt_mm(W, iters=200, tol=1e-8); cov = bt_fisher_cov(beta, W)
            # greedy: among all pairs, the one whose next comparison most reduces the
            # largest gap variance among the contested set (one-step look-ahead)
            best, best_val = None, -1.0
            for (a, b) in pairs:
                p = 1 / (1 + math.exp(beta[b] - beta[a])); v = p * (1 - p)
                d = np.zeros(K); d[a] = 1; d[b] = -1
                # Sherman-Morrison: new cov = cov - v cov d d' cov / (1 + v d' cov d)
                cd = cov @ d; denom = 1 + v * (d @ cd)
                gain = 0.0
                for (c1, c2) in [(x, y) for x in contested for y in contested if x < y]:
                    e = np.zeros(K); e[c1] = 1; e[c2] = -1
                    gain = max(gain, v * (e @ cd) ** 2 / denom)
                if gain > best_val:
                    best_val, best = gain, (a, b)
            i, j = best
        out = simulate_battle(i, j, rng)
        if out == 0: W[i, j] += 1
        elif out == 1: W[j, i] += 1
        else: W[i, j] += 0.5; W[j, i] += 0.5
        if (t + 1) % every == 0 and t + 1 >= warm:
            beta, _, _ = bt_mm(W, iters=200, tol=1e-8); cov = bt_fisher_cov(beta, W)
            hist.append(max(se_diff(cov, a, b) for a in contested for b in contested if a < b))
    return np.array(hist)


def main():
    num = {"z": Z, "true_beta": TRUE_BETA.tolist(), "models": MODELS}
    rng = np.random.default_rng(11)
    df = pd.read_csv(ROOT / "data" / "arena300.csv")
    n = len(df); num["n_battles"] = n
    num["outcomes"] = df["outcome"].value_counts().to_dict()
    W = win_matrix(df); N = W + W.T
    num["pair_counts"] = {f"{MODELS[i]}-{MODELS[j]}": int(N[i, j]) for i in range(K) for j in range(i + 1, K)}
    num["min_pair"] = int(N[np.triu_indices(K, 1)].min()); num["max_pair"] = int(N[np.triu_indices(K, 1)].max())
    num["wins"] = {m: float(W[k].sum()) for k, m in enumerate(MODELS)}
    num["games"] = {m: float(N[k].sum()) for k, m in enumerate(MODELS)}
    num["win_rate"] = {m: float(W[k].sum() / N[k].sum()) for k, m in enumerate(MODELS)}

    # --- 11.2 MM fit -----------------------------------------------------------------
    beta, iters, ll = bt_mm(W)
    cov = bt_fisher_cov(beta, W)
    num["mm"] = {"beta": beta.tolist(), "iters": iters, "loglik": ll, "se": np.sqrt(np.diag(cov)).tolist(),
                 "order": [MODELS[k] for k in (-beta).argsort()]}
    num["mm"]["gaps"] = {}
    order = (-beta).argsort()
    for a in range(K - 1):
        i, j = order[a], order[a + 1]
        num["mm"]["gaps"][f"{MODELS[i]}-{MODELS[j]}"] = {"gap": float(beta[i] - beta[j]), "se": se_diff(cov, i, j),
                                                         "z": float((beta[i] - beta[j]) / se_diff(cov, i, j))}
    num["mm"]["gap_M2_M5"] = {"gap": float(beta[1] - beta[4]), "se": se_diff(cov, 1, 4)}
    num["mm"]["gap_M1_M2"] = {"gap": float(beta[0] - beta[1]), "se": se_diff(cov, 0, 1)}
    num["mm"]["gap_M5_M6"] = {"gap": float(beta[4] - beta[5]), "se": se_diff(cov, 4, 5)}
    # win probabilities implied
    num["mm"]["p_M1_beats_M2"] = float(1 / (1 + math.exp(beta[1] - beta[0])))
    num["mm"]["p_M2_beats_M5"] = float(1 / (1 + math.exp(beta[4] - beta[1])))
    # MM iteration count at a looser tolerance, for the text
    num["mm"]["iters_1e-6"] = bt_mm(W, tol=1e-6)[1]

    # --- 11.3 logistic form, with and without the position term ---------------------------
    X, y = design(df)
    b_noposition, _, cov_np, ll_np = logistic_fit(X, y, drop_position=True)
    b_pos, gamma, cov_pos, ll_pos = logistic_fit(X, y)
    assert np.max(np.abs(b_noposition - beta)) < 1e-5, (b_noposition, beta)
    num["logistic"] = {"beta_noposition": b_noposition.tolist(), "loglik_noposition": ll_np,
                       "beta_position": b_pos.tolist(), "gamma": float(gamma), "gamma_se": float(math.sqrt(cov_pos[K - 1, K - 1])),
                       "loglik_position": ll_pos, "lr_stat": 2 * (ll_pos - ll_np),
                       "order_position": [MODELS[k] for k in (-b_pos).argsort()],
                       "first_win_share": float((df.outcome == "first").mean() / (df.outcome != "tie").mean())}
    # statsmodels cross-check of the position model (ties as 0.5 via GLM Binomial)
    try:
        import statsmodels.api as sm
        cols = list(range(K - 1)) + [K]
        Xr = X[:, cols] - X[:, K - 1][:, None] * np.array([1.0] * (K - 1) + [0.0])
        glm = sm.GLM(y, Xr, family=sm.families.Binomial()).fit()
        assert np.max(np.abs(glm.params[:K - 1] - b_pos[:K - 1])) < 1e-4
        assert abs(glm.params[K - 1] - gamma) < 1e-4
        num["logistic"]["statsmodels_gamma_se"] = float(glm.bse[K - 1])
    except ImportError:
        pass

    # --- 11.4 Elo in three orders -----------------------------------------------------
    orders = {"as_played": None, "reversed": list(range(n))[::-1], "shuffled": list(rng.permutation(n))}
    num["elo"] = {}
    for name, o in orders.items():
        R = elo(df, order=o)
        num["elo"][name] = {"ratings": {m: float(R[m]) for m in MODELS}, "order": sorted(MODELS, key=lambda m: -R[m])}
    # Elo scale: beta = R ln 10 / 400
    num["elo"]["scale"] = math.log(10) / 400
    num["elo"]["bt_as_elo"] = {m: float(1000 + beta[k] * 400 / math.log(10)) for k, m in enumerate(MODELS)}
    # spread across 200 random orders
    finals = np.array([[elo(df, order=list(rng.permutation(n)))[m] for m in MODELS] for _ in range(200)])
    num["elo"]["order_sd"] = {m: float(finals[:, k].std(ddof=1)) for k, m in enumerate(MODELS)}
    num["elo"]["top_agree"] = float(np.mean(finals.argmax(1) == 0))
    ranks = (-finals).argsort(1).argsort(1) + 1
    num["elo"]["rank_of_M2_range"] = [int(ranks[:, 1].min()), int(ranks[:, 1].max())]
    num["elo"]["rank_of_M5_range"] = [int(ranks[:, 4].min()), int(ranks[:, 4].max())]
    # small K: closer to BT
    R8 = elo(df, k_factor=8)
    num["elo"]["k8_order"] = sorted(MODELS, key=lambda m: -R8[m])

    # --- 11.5 bootstrap over battles ----------------------------------------------------
    B = 2000
    betas, ranks = bootstrap_bt(df, B, rng)
    num["boot"] = {"reps": B, "se": betas.std(0, ddof=1).tolist(),
                   "lo": np.quantile(betas, 0.025, axis=0).tolist(), "hi": np.quantile(betas, 0.975, axis=0).tolist(),
                   "rank_lo": np.quantile(ranks, 0.025, axis=0).astype(int).tolist(),
                   "rank_hi": np.quantile(ranks, 0.975, axis=0).astype(int).tolist(),
                   "p_first": (ranks == 1).mean(0).tolist(), "p_second": (ranks == 2).mean(0).tolist()}
    num["boot"]["plausibly_best"] = [m for k, m in enumerate(MODELS) if num["boot"]["p_first"][k] > 0.025]
    num["boot"]["plausibly_second"] = [m for k, m in enumerate(MODELS) if num["boot"]["p_second"][k] > 0.025]
    gap25 = betas[:, 1] - betas[:, 4]
    num["boot"]["gap_M2_M5_lo"] = float(np.quantile(gap25, 0.025)); num["boot"]["gap_M2_M5_hi"] = float(np.quantile(gap25, 0.975))
    num["boot"]["gap_M2_M5_se"] = float(gap25.std(ddof=1))

    # --- 11.6 ties: Davidson model fit ---------------------------------------------------
    idx = {m: k for k, m in enumerate(MODELS)}
    trip = [(idx[r["first"]], idx[r["second"]], r["outcome"]) for _, r in df.iterrows()]

    def davidson_nll(th):
        b = np.append(th[:K - 1], -th[:K - 1].sum()); lnu = th[K - 1]
        ll = 0.0
        for i, j, o in trip:
            li, lj = b[i], b[j]; lt = lnu + 0.5 * (li + lj)
            lse = np.logaddexp(np.logaddexp(li, lj), lt)
            ll += {"first": li, "second": lj, "tie": lt}[o] - lse
        return -ll

    res = minimize(davidson_nll, np.zeros(K), method="BFGS")
    bd = np.append(res.x[:K - 1], -res.x[:K - 1].sum())
    num["davidson"] = {"beta": bd.tolist(), "nu": float(math.exp(res.x[K - 1])), "loglik": -res.fun,
                       "order": [MODELS[k] for k in (-bd).argsort()], "true_nu": 0.25,
                       "max_abs_diff_vs_halfwin": float(np.max(np.abs(bd - beta)))}

    # --- 11.7 how many battles? Fisher-information planning and active selection -------------
    # expected SE of a gap between two equal models with m comparisons between them, isolated pair
    num["plan"] = {"se_gap_direct_m": {str(m): float(math.sqrt(4 / m)) for m in (20, 50, 100, 200, 400)},
                   "m_for_gap_0.15_at_z2": float(4 * (2 / 0.15) ** 2)}
    # simulation: uniform vs greedy, 400 battles, 20 replications; SE of the largest contested gap
    reps = 20; n_total = 400; every = 10
    hist_u = np.array([run_schedule("uniform", n_total, rng, every=every) for _ in range(reps)])
    hist_g = np.array([run_schedule("greedy", n_total, rng, every=every) for _ in range(reps)])
    checkpoints = list(range(60, n_total + 1, every))
    num["active"] = {"checkpoints": checkpoints, "uniform_mean_se": hist_u.mean(0).tolist(),
                     "greedy_mean_se": hist_g.mean(0).tolist(), "reps": reps, "n_total": n_total}
    for c in (100, 200, 300, 400):
        k = checkpoints.index(c)
        num["active"][f"se_at_{c}"] = {"uniform": float(hist_u[:, k].mean()), "greedy": float(hist_g[:, k].mean())}
    # battles the greedy scheme needs to reach the uniform scheme's SE at 400
    target = hist_u[:, -1].mean()
    reach = next((checkpoints[k] for k in range(len(checkpoints)) if hist_g[:, k].mean() <= target), None)
    num["active"]["greedy_reaches_uniform400_at"] = reach

    OUT.write_text(json.dumps(num, indent=2))
    for k, v in num.items():
        print(k, json.dumps(v, default=str)[:1200])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
