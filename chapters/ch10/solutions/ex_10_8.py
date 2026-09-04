"""Exercise 10.8: tune the CUSUM, compare monitors, and mis-specify p1."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from worked_example import cusum_arl0, cusum_alarm, beta_binomial_cs
from scipy.stats import norm
P0, N, CHANGE, DAYS = 0.80, 200, 40, 100
rng = np.random.default_rng(108)

def tune(p1, target=700.0):
    hs = [2, 3, 4, 5, 6, 7]; la = [math.log(cusum_arl0(h, N, P0, p1, 300, rng, horizon=4000)[0]) for h in hs]
    return float(np.interp(math.log(target), la, hs))

alt = np.concatenate([rng.binomial(N, P0, size=(1000, CHANGE - 1)), rng.binomial(N, 0.75, size=(1000, DAYS - CHANGE + 1))], 1)
null = rng.binomial(N, P0, size=(1000, DAYS))
for p1 in (0.75, 0.78):
    h = tune(p1)
    fa = np.mean([cusum_alarm(c, N, P0, p1, h) is not None for c in null])
    al = [cusum_alarm(c, N, P0, p1, h) for c in alt]
    delays = [a - CHANGE + 1 for a in al if a is not None and a >= CHANGE]
    print(f"CUSUM p1 = {p1}: h = {h:.2f}, false alarm in 100 days {fa:.3f}, detected {np.mean([a is not None and a >= CHANGE for a in al]):.3f}, "
          f"mean delay {np.mean(delays):.1f}, 90th pct {np.quantile(delays, 0.9):.0f}")
# confidence sequence and daily test for comparison
NN = N * np.arange(1, DAYS + 1); zc = norm.ppf(0.95)
def cs_alarm(c):
    S = c.cumsum()
    return next((t + 1 for t in range(DAYS) if beta_binomial_cs(int(S[t]), int(NN[t]), 0.05)[1] < P0), None)
def peek_alarm(c):
    S = c.cumsum(); z = (S - NN * P0) / np.sqrt(NN * P0 * (1 - P0)); hit = np.where(z < -zc)[0]
    return int(hit[0]) + 1 if hit.size else None
for name, f in (("confidence sequence", cs_alarm), ("daily cumulative test", peek_alarm)):
    fa = np.mean([f(c) is not None for c in null]); al = [f(c) for c in alt]
    delays = [a - CHANGE + 1 for a in al if a is not None and a >= CHANGE]
    print(f"{name}: false alarm {fa:.3f}, detected {np.mean([a is not None and a >= CHANGE for a in al]):.3f}, mean delay {np.mean(delays):.1f}")
