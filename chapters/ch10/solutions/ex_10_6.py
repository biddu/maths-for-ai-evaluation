"""Exercise 10.6: the peeking problem, one- and two-sided, daily and weekly; Bonferroni over 100 looks."""
import numpy as np
from scipy.stats import norm
P0, P1, N, CHANGE = 0.80, 0.75, 200, 40
rng = np.random.default_rng(106)
reps, days = 4000, 365
counts = rng.binomial(N, P0, size=(reps, days)); S = counts.cumsum(1); NN = N * np.arange(1, days + 1)
z = (S - NN * P0) / np.sqrt(NN * P0 * (1 - P0))
one = np.minimum.accumulate(z, axis=1) < -norm.ppf(0.95)
two = np.maximum.accumulate(np.abs(z), axis=1) > norm.ppf(0.975)
weekly = z[:, 6::7]
print("T        1     7    14    30   100   365")
print("one-sided", [round(one[:, T - 1].mean(), 3) for T in (1, 7, 14, 30, 100, 365)])
print("two-sided", [round(two[:, T - 1].mean(), 3) for T in (1, 7, 14, 30, 100, 365)])
print("weekly looks, one-sided, by week:", [round((weekly[:, :w] < -norm.ppf(0.95)).any(1).mean(), 3) for w in (1, 4, 14, 52)])
zb = norm.ppf(1 - 0.05 / 100)
print(f"Bonferroni over 100 daily looks: false alarm {(z[:, :100] < -zb).any(1).mean():.4f}")
alt = np.concatenate([rng.binomial(N, P0, size=(reps, CHANGE - 1)), rng.binomial(N, P1, size=(reps, 100 - CHANGE + 1))], 1)
Sa = alt.cumsum(1); Na = N * np.arange(1, 101); za = (Sa - Na * P0) / np.sqrt(Na * P0 * (1 - P0))
hit = za < -zb
first = np.where(hit.any(1), hit.argmax(1) + 1, 0)
print(f"Bonferroni power by day 100 (alarm on/after day 40): {((first >= CHANGE)).mean():.3f}, false alarm before day 40: {((first > 0) & (first < CHANGE)).mean():.4f}, mean delay {np.mean(first[first >= CHANGE] - CHANGE + 1):.1f}")
