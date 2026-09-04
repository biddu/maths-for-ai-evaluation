"""Exercise 10.7: the confidence sequence with two priors, from day 1 and from day 40."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from worked_example import beta_binomial_cs, ROOT
d = pd.read_csv(ROOT / "data" / "daily_stream.csv"); daily = d.groupby("day")["pass"].sum().to_numpy()
for start in (1, 40):
    S = daily[start - 1:].cumsum(); N = 200 * np.arange(1, len(S) + 1)
    for prior in ((1, 1), (8, 2)):
        cs = [beta_binomial_cs(int(S[t]), int(N[t]), 0.05, *prior) for t in range(len(S))]
        first = next((t + 1 for t, (lo, hi) in enumerate(cs) if hi < 0.80), None)
        shown = {t: tuple(round(x, 3) for x in cs[t - 1]) for t in (1, 7, 39) if t <= len(cs)}
        print(f"start day {start}, Beta{prior}: {shown}, first day excluding 0.80: {first}")
