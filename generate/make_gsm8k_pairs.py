"""Generate gsm8k_pairs: per-item correctness for two models on 1,319 items.

Synthetic, with the structure of real paired benchmark output: a shared
per-item difficulty makes hard items hard for both models, so the two
outcome columns are positively correlated. Model C is slightly stronger
than model D.

The seed is chosen so that the example in Chapter 3 makes its point:
the paired test finds the difference at the 5% level and the (wrong)
unpaired test does not.

Output: data/gsm8k_pairs.csv with columns item, C, D  (0/1 each)
"""
import csv
import math
import pathlib
import numpy as np
from scipy.stats import binomtest, norm

N = 1319
OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "gsm8k_pairs.csv"


def draw(seed):
    rng = np.random.default_rng(seed)
    shared = rng.normal(2.6, 3.2, N)
    lc = shared + rng.normal(0.22, 0.5, N)
    ld = shared + rng.normal(0.00, 0.5, N)
    pc, pd = 1 / (1 + np.exp(-lc)), 1 / (1 + np.exp(-ld))
    return (rng.random(N) < pc).astype(int), (rng.random(N) < pd).astype(int)


def tests(c, d):
    b = int(((c == 1) & (d == 0)).sum())
    cc = int(((c == 0) & (d == 1)).sum())
    p_exact = binomtest(b, b + cc, 0.5).pvalue
    pa, pb = c.mean(), d.mean()
    se_unp = math.sqrt((pa * (1 - pa) + pb * (1 - pb)) / N)
    p_unp = 2 * (1 - norm.cdf(abs(pa - pb) / se_unp))
    return b, cc, p_exact, p_unp, pa, pb


def main():
    for seed in range(20_000):
        c, d = draw(seed)
        b, cc, p_exact, p_unp, pa, pb = tests(c, d)
        gap = pa - pb
        if 0.018 <= gap <= 0.024 and p_exact < 0.045 and p_unp > 0.10 and 0.78 <= pa <= 0.82:
            break
    else:
        raise SystemExit("no seed found")
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "C", "D"])
        for i in range(N):
            w.writerow([i, int(c[i]), int(d[i])])
    print(f"seed={seed} C={c.sum()} D={d.sum()} b={b} c={cc} p_exact={p_exact:.4f} p_unpaired={p_unp:.3f} -> {OUT}")


if __name__ == "__main__":
    main()
