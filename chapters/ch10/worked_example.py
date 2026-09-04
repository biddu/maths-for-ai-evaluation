"""Chapter 10 -- every number printed in the chapter, regenerated from data.

Run:  python chapters/ch10/worked_example.py
Writes chapters/ch10/numbers.json.

Data: data/daily_stream.csv (model N, 200 items a day for 100 days, true pass
rate 0.80 until day 39 and 0.75 from day 40).
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import betaln
from scipy.optimize import brentq

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).with_name("numbers.json")
Z = norm.ppf(0.975)
P0, P1, PER_DAY, CHANGE_DAY, DAYS = 0.80, 0.75, 200, 40, 100
ALPHA, BETA = 0.05, 0.10


# ---------------------------------------------------------------------------
# 10.1 peeking: a daily one-sided z-test against a known baseline
# ---------------------------------------------------------------------------
def peeking_sim(days, n_per_day, p0, alpha, reps, rng):
    """Fraction of null streams in which the cumulative z-test rejects on at least one day."""
    z_crit = norm.ppf(1 - alpha)
    counts = rng.binomial(n_per_day, p0, size=(reps, days))
    S = counts.cumsum(1); N = n_per_day * np.arange(1, days + 1)
    z = (S - N * p0) / np.sqrt(N * p0 * (1 - p0))
    return float((z < -z_crit).any(1).mean()), (z < -z_crit)


# ---------------------------------------------------------------------------
# 10.2 Wald's SPRT for Bernoulli
# ---------------------------------------------------------------------------
def llr_increment(p0, p1):
    return math.log(p1 / p0), math.log((1 - p1) / (1 - p0))


def sprt(outcomes, p0, p1, alpha, beta):
    """outcomes: 0/1 items in order. Returns (decision, n) with decision in {'H0','H1',None}."""
    a, b = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    lp, lf = llr_increment(p0, p1)
    L = 0.0
    for i, x in enumerate(outcomes, 1):
        L += lp if x else lf
        if L >= a:
            return "H1", i
        if L <= b:
            return "H0", i
    return None, len(outcomes)


def sprt_expected_n(p, p0, p1, alpha, beta):
    """Wald's approximation to E_p[N], ignoring overshoot."""
    a, b = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    lp, lf = llr_increment(p0, p1)
    drift = p * lp + (1 - p) * lf
    if p == p0:
        return ((1 - alpha) * b + alpha * a) / drift
    if p == p1:
        return (beta * b + (1 - beta) * a) / drift
    # general p: probability of accepting H1 from Wald's approximation with h such that E[exp(h Z)] = 1
    raise ValueError("only p0 or p1")


def fixed_n_one_sided(p0, p1, alpha, beta):
    za, zb = norm.ppf(1 - alpha), norm.ppf(1 - beta)
    return (za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))) ** 2 / (p1 - p0) ** 2


# ---------------------------------------------------------------------------
# 10.3 Beta-binomial mixture martingale and confidence sequence
# ---------------------------------------------------------------------------
def log_mixture(S, n, p, a=1.0, b=1.0):
    """log M_n(p): Beta(a,b)-mixture likelihood over the null p."""
    return betaln(a + S, b + n - S) - betaln(a, b) - S * math.log(p) - (n - S) * math.log(1 - p)


def beta_binomial_cs(S, n, alpha, a=1.0, b=1.0):
    """Confidence sequence: all p with M_n(p) < 1/alpha. Returns (lo, hi)."""
    thr = math.log(1 / alpha)
    f = lambda p: log_mixture(S, n, p, a, b) - thr
    phat = (S + a) / (n + a + b)
    eps = 1e-9
    lo = brentq(f, eps, phat) if f(eps) > 0 else 0.0
    hi = brentq(f, phat, 1 - eps) if f(1 - eps) > 0 else 1.0
    return lo, hi


# ---------------------------------------------------------------------------
# 10.5 CUSUM
# ---------------------------------------------------------------------------
def cusum_path(daily_counts, n_per_day, p0, p1):
    lp, lf = llr_increment(p0, p1)
    S = 0.0; path = []
    for c in daily_counts:
        z = c * lp + (n_per_day - c) * lf
        S = max(0.0, S + z); path.append(S)
    return np.array(path)


def cusum_alarm(daily_counts, n_per_day, p0, p1, h):
    path = cusum_path(daily_counts, n_per_day, p0, p1)
    hits = np.where(path >= h)[0]
    return int(hits[0]) + 1 if hits.size else None


def cusum_arl0(h, n_per_day, p0, p1, reps, rng, horizon=5000):
    """Average run length under the null by simulation (censored at horizon)."""
    lp, lf = llr_increment(p0, p1)
    counts = rng.binomial(n_per_day, p0, size=(reps, horizon))
    z = counts * lp + (n_per_day - counts) * lf
    run = np.empty(reps)
    for r in range(reps):
        S = 0.0
        for t in range(horizon):
            S = max(0.0, S + z[r, t])
            if S >= h:
                run[r] = t + 1; break
        else:
            run[r] = horizon
    return float(run.mean()), float(np.median(run)), float((run < horizon).mean())


def main():
    num = {"z": Z, "p0": P0, "p1": P1, "per_day": PER_DAY, "change_day": CHANGE_DAY, "days": DAYS,
           "alpha": ALPHA, "beta": BETA}
    rng = np.random.default_rng(10)
    d = pd.read_csv(ROOT / "data" / "daily_stream.csv")
    daily = d.groupby("day")["pass"].sum().to_numpy()
    num["mean_before"] = float(daily[:CHANGE_DAY - 1].sum() / (PER_DAY * (CHANGE_DAY - 1)))
    num["mean_after"] = float(daily[CHANGE_DAY - 1:].sum() / (PER_DAY * (DAYS - CHANGE_DAY + 1)))

    # --- 10.1 peeking inflation ---------------------------------------------------------
    reps = 4000
    num["peek"] = {}
    for T in (1, 7, 14, 30, 100, 365):
        num["peek"][str(T)] = peeking_sim(T, PER_DAY, P0, ALPHA, reps, rng)[0]
    num["peek_reps"] = reps
    # two-sided version at 0.05 for the failure box (the dashboard's p-value)
    counts = rng.binomial(PER_DAY, P0, size=(reps, 100))
    S = counts.cumsum(1); N = PER_DAY * np.arange(1, 101)
    z = (S - N * P0) / np.sqrt(N * P0 * (1 - P0))
    num["peek_two_sided"] = {str(T): float((np.abs(z[:, :T]) > Z).any(1).mean()) for T in (7, 30, 100)}
    # the day-by-day version: probability the FIRST rejection is on day t (null)
    first = np.argmax(np.abs(z) > Z, axis=1) + 1; first[~(np.abs(z) > Z).any(1)] = 0
    num["peek_first_day_median"] = float(np.median(first[first > 0]))

    # --- 10.2 SPRT: thresholds, expected sample sizes, and simulated error rates ------------
    a, b = math.log((1 - BETA) / ALPHA), math.log(BETA / (1 - ALPHA))
    lp, lf = llr_increment(P0, P1)
    num["sprt"] = {"logA": a, "logB": b, "A": (1 - BETA) / ALPHA, "B": BETA / (1 - ALPHA),
                   "inc_pass": lp, "inc_fail": lf,
                   "drift0": P0 * lp + (1 - P0) * lf, "drift1": P1 * lp + (1 - P1) * lf,
                   "EN0": sprt_expected_n(P0, P0, P1, ALPHA, BETA), "EN1": sprt_expected_n(P1, P0, P1, ALPHA, BETA),
                   "fixed_n": fixed_n_one_sided(P0, P1, ALPHA, BETA)}
    # simulate the SPRT as a deployment-day decision, item by item, 4000 streams each
    sim_reps = 4000
    for label, p in (("null", P0), ("alt", P1)):
        dec, ns = [], []
        for _ in range(sim_reps):
            x = rng.random(4000) < p
            dcs, n = sprt(x, P0, P1, ALPHA, BETA)
            dec.append(dcs); ns.append(n)
        dec = np.array(dec); ns = np.array(ns)
        num["sprt"][f"{label}_reject"] = float((dec == "H1").mean())
        num["sprt"][f"{label}_undecided"] = float((dec == None).mean())  # noqa: E711
        num["sprt"][f"{label}_mean_n"] = float(ns.mean()); num["sprt"][f"{label}_median_n"] = float(np.median(ns))
        num["sprt"][f"{label}_q90_n"] = float(np.quantile(ns, 0.9))
    # fixed-sample test at the same n, for comparison: one-sided z at n = fixed_n
    nf = int(math.ceil(num["sprt"]["fixed_n"]))
    num["sprt"]["fixed_n_int"] = nf
    for label, p in (("null", P0), ("alt", P1)):
        c = rng.binomial(nf, p, size=sim_reps)
        zz = (c / nf - P0) / math.sqrt(P0 * (1 - P0) / nf)
        num["sprt"][f"fixed_{label}_reject"] = float((zz < -norm.ppf(1 - ALPHA)).mean())
    # the SPRT on the actual stream, started on day 40 (deployment-day decision) and on day 1
    items_from_40 = d[d.day >= CHANGE_DAY].sort_values(["day", "item"])["pass"].to_numpy()
    dcs, n = sprt(items_from_40, P0, P1, ALPHA, BETA)
    num["sprt"]["data_from_40"] = {"decision": dcs, "n": n, "day": CHANGE_DAY + (n - 1) // PER_DAY}
    items_from_1 = d.sort_values(["day", "item"])["pass"].to_numpy()
    dcs, n = sprt(items_from_1, P0, P1, ALPHA, BETA)
    num["sprt"]["data_from_1"] = {"decision": dcs, "n": n, "day": 1 + (n - 1) // PER_DAY}

    # --- 10.3 confidence sequence on the data, day by day ------------------------------------
    S = daily.cumsum(); N = PER_DAY * np.arange(1, DAYS + 1)
    cs = np.array([beta_binomial_cs(int(S[t]), int(N[t]), ALPHA) for t in range(DAYS)])
    wilson_hw = Z * np.sqrt((S / N) * (1 - S / N) / N)
    num["cs"] = {"day": [1, 7, 30, 39, 60, 100]}
    for t in num["cs"]["day"]:
        num["cs"][f"d{t}"] = {"phat": float(S[t - 1] / N[t - 1]), "lo": float(cs[t - 1, 0]), "hi": float(cs[t - 1, 1]),
                              "width": float(cs[t - 1, 1] - cs[t - 1, 0]), "wilson_width": float(2 * wilson_hw[t - 1]),
                              "ratio": float((cs[t - 1, 1] - cs[t - 1, 0]) / (2 * wilson_hw[t - 1]))}
    excl = np.where(cs[:, 1] < P0)[0]
    num["cs"]["first_day_excluding_p0"] = int(excl[0]) + 1 if excl.size else None
    # CS on a stream that starts at the change (deployment-day decision)
    S40 = daily[CHANGE_DAY - 1:].cumsum(); N40 = PER_DAY * np.arange(1, DAYS - CHANGE_DAY + 2)
    cs40 = np.array([beta_binomial_cs(int(S40[t]), int(N40[t]), ALPHA) for t in range(len(S40))])
    excl40 = np.where(cs40[:, 1] < P0)[0]
    num["cs"]["from_40_first_day_excluding_p0"] = int(excl40[0]) + 1 if excl40.size else None
    # the martingale M_n(p0) on the data from day 40 and from day 1 (as an e-value)
    num["cs"]["logM_p0_from40"] = [float(log_mixture(int(S40[t]), int(N40[t]), P0)) for t in range(len(S40))]
    num["cs"]["logM_p0_from1"] = [float(log_mixture(int(S[t]), int(N[t]), P0)) for t in range(DAYS)]
    # Ville check: fraction of null streams (100 days) in which M_n(p0) ever exceeds 1/alpha
    counts = rng.binomial(PER_DAY, P0, size=(2000, DAYS)); Sn = counts.cumsum(1)
    ever = np.zeros(2000, bool)
    for r in range(2000):
        for t in range(DAYS):
            if log_mixture(int(Sn[r, t]), int(N[t]), P0) >= math.log(1 / ALPHA):
                ever[r] = True; break
    num["cs"]["ville_100_days"] = float(ever.mean())
    # the same for 1000 days, coarser (weekly checks are enough since M is checked at every n anyway: use daily)
    counts = rng.binomial(PER_DAY, P0, size=(1000, 1000)); Sn = counts.cumsum(1); Nn = PER_DAY * np.arange(1, 1001)
    ever = np.zeros(1000, bool)
    for r in range(1000):
        lm = [log_mixture(int(Sn[r, t]), int(Nn[t]), P0) for t in range(1000)]
        ever[r] = max(lm) >= math.log(1 / ALPHA)
    num["cs"]["ville_1000_days"] = float(ever.mean())

    # --- 10.5 CUSUM: tune h for a weekly false-alarm rate of 1% (ARL0 about 700 days) ------------
    num["cusum"] = {}
    target_arl0 = 700.0
    h_grid = [2, 3, 4, 5, 6, 7, 8]
    arls = {}
    for h in h_grid:
        arl, med, frac = cusum_arl0(h, PER_DAY, P0, P1, 400, rng, horizon=4000)
        arls[h] = arl
    num["cusum"]["arl0_by_h"] = {str(h): v for h, v in arls.items()}
    # interpolate log ARL0 in h to hit the target
    hs = np.array(h_grid, float); la = np.log(np.array([arls[h] for h in h_grid]))
    h_star = float(np.interp(math.log(target_arl0), la, hs))
    num["cusum"]["h_star"] = h_star
    arl0, med0, frac0 = cusum_arl0(h_star, PER_DAY, P0, P1, 1000, rng, horizon=6000)
    num["cusum"]["arl0_at_h_star"] = arl0; num["cusum"]["arl0_median"] = med0
    num["cusum"]["weekly_false_alarm"] = 1 - math.exp(-7 / arl0)
    # detection delay under the alternative (drop from day 1 of the monitored stream)
    counts = rng.binomial(PER_DAY, P1, size=(2000, 60))
    delays = np.array([cusum_alarm(c, PER_DAY, P0, P1, h_star) or 61 for c in counts])
    num["cusum"]["arl1_mean"] = float(delays.mean()); num["cusum"]["arl1_median"] = float(np.median(delays))
    num["cusum"]["arl1_q90"] = float(np.quantile(delays, 0.9))
    # on the data
    path = cusum_path(daily, PER_DAY, P0, P1)
    num["cusum"]["path"] = path.tolist()
    num["cusum"]["data_alarm_day"] = cusum_alarm(daily, PER_DAY, P0, P1, h_star)
    num["cusum"]["path_max_before_change"] = float(path[:CHANGE_DAY - 1].max())

    # --- 10.6 the monitoring comparison over 1,000 replications ---------------------------------
    R = 1000
    null_counts = rng.binomial(PER_DAY, P0, size=(R, DAYS))
    alt_counts = np.concatenate([rng.binomial(PER_DAY, P0, size=(R, CHANGE_DAY - 1)),
                                 rng.binomial(PER_DAY, P1, size=(R, DAYS - CHANGE_DAY + 1))], axis=1)
    z_crit = norm.ppf(1 - ALPHA)

    def monitor(counts):
        out = {"peek": [], "cs": [], "cusum": [], "peek_window": []}
        for c in counts:
            Sc = c.cumsum(); z = (Sc - N * P0) / np.sqrt(N * P0 * (1 - P0))
            hit = np.where(z < -z_crit)[0]; out["peek"].append(int(hit[0]) + 1 if hit.size else None)
            # a 7-day moving-window z-test, tested daily (a common dashboard design)
            win = np.array([c[max(0, t - 6):t + 1].sum() for t in range(DAYS)])
            nw = np.array([PER_DAY * min(7, t + 1) for t in range(DAYS)])
            zw = (win - nw * P0) / np.sqrt(nw * P0 * (1 - P0))
            hit = np.where(zw < -z_crit)[0]; out["peek_window"].append(int(hit[0]) + 1 if hit.size else None)
            hit = None
            for t in range(DAYS):
                if beta_binomial_cs(int(Sc[t]), int(N[t]), ALPHA)[1] < P0:
                    hit = t + 1; break
            out["cs"].append(hit)
            out["cusum"].append(cusum_alarm(c, PER_DAY, P0, P1, h_star))
        return out

    null_out = monitor(null_counts); alt_out = monitor(alt_counts)
    num["monitor"] = {}
    for m in ("peek", "peek_window", "cs", "cusum"):
        fa = np.array([x is not None for x in null_out[m]])
        det = np.array([x is not None and x >= CHANGE_DAY for x in alt_out[m]])
        early = np.array([x is not None and x < CHANGE_DAY for x in alt_out[m]])
        delays = np.array([x - CHANGE_DAY + 1 for x in alt_out[m] if x is not None and x >= CHANGE_DAY])
        num["monitor"][m] = {"false_alarm_100d": float(fa.mean()),
                             "false_alarm_before_change_alt": float(early.mean()),
                             "detected_by_100": float(det.mean()),
                             "delay_mean": float(delays.mean()) if delays.size else None,
                             "delay_median": float(np.median(delays)) if delays.size else None,
                             "delay_q90": float(np.quantile(delays, 0.9)) if delays.size else None}
    num["monitor"]["reps"] = R
    # on the actual data: alarm days for each monitor
    Sc = daily.cumsum(); z = (Sc - N * P0) / np.sqrt(N * P0 * (1 - P0))
    hit = np.where(z < -z_crit)[0]
    num["monitor"]["data"] = {"peek_first_day": int(hit[0]) + 1 if hit.size else None,
                              "peek_days_flagged": int((z < -z_crit).sum()),
                              "z_path": z.tolist(), "cs_lo": cs[:, 0].tolist(), "cs_hi": cs[:, 1].tolist(),
                              "cs_first_day": num["cs"]["first_day_excluding_p0"],
                              "cusum_day": num["cusum"]["data_alarm_day"]}
    win = np.array([daily[max(0, t - 6):t + 1].sum() for t in range(DAYS)])
    nw = np.array([PER_DAY * min(7, t + 1) for t in range(DAYS)])
    zw = (win - nw * P0) / np.sqrt(nw * P0 * (1 - P0)); hit = np.where(zw < -z_crit)[0]
    num["monitor"]["data"]["peek_window_first_day"] = int(hit[0]) + 1 if hit.size else None
    num["monitor"]["data"]["peek_window_flags_before_change"] = int((zw[:CHANGE_DAY - 1] < -z_crit).sum())

    # --- 10.3 the two-sample case: a sequential McNemar on discordant pairs --------------------
    # Paired stream: each day both models see 200 items; a pair is discordant with prob 0.15,
    # and a discordant pair favours A with probability pi (0.5 under H0, 0.6 under H1).
    disc_rate = 0.15
    num["paired"] = {"disc_rate": disc_rate, "pi1": 0.6,
                     "fixed_discordant": fixed_n_one_sided(0.5, 0.6, ALPHA, BETA)}
    num["paired"]["fixed_items"] = num["paired"]["fixed_discordant"] / disc_rate
    num["paired"]["fixed_days"] = num["paired"]["fixed_items"] / PER_DAY
    R2 = 1000
    for label, pi in (("null", 0.5), ("alt", 0.6)):
        decide, wrong = [], 0
        for _ in range(R2):
            m = rng.binomial(PER_DAY, disc_rate, size=DAYS).cumsum()      # discordant pairs so far
            bA = np.array([rng.binomial(int(mi), pi) for mi in np.diff(np.concatenate([[0], m]))]).cumsum()
            day = None
            for t in range(DAYS):
                if m[t] == 0:
                    continue
                lo, hi = beta_binomial_cs(int(bA[t]), int(m[t]), ALPHA)
                if lo > 0.5:
                    day = t + 1; break
                if hi < 0.5:
                    day = t + 1; wrong += 1; break
            decide.append(day)
        dec = np.array([x for x in decide if x is not None])
        num["paired"][f"{label}_decided_100"] = float(np.mean([x is not None for x in decide]))
        num["paired"][f"{label}_mean_day"] = float(dec.mean()) if dec.size else None
        num["paired"][f"{label}_median_day"] = float(np.median(dec)) if dec.size else None
        num["paired"][f"{label}_q90_day"] = float(np.quantile(dec, 0.9)) if dec.size else None
        num["paired"][f"{label}_wrong_direction"] = wrong / R2

    OUT.write_text(json.dumps(num, indent=2))
    for k, v in num.items():
        if isinstance(v, dict):
            print(k, json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()
                                 if not (isinstance(vv, list) and len(vv) > 10)}, default=str)[:1800])
        else:
            print(k, v)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
