"""Exercise 11.7: Elo order-dependence at three step sizes."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from worked_example import win_matrix, bt_mm, elo, MODELS, ROOT
df = pd.read_csv(ROOT / "data" / "arena300.csv"); n = len(df)
beta, _, _ = bt_mm(win_matrix(df)); bt_elo = 1000 + beta * 400 / math.log(10)
rng = np.random.default_rng(117)
orders = [list(rng.permutation(n)) for _ in range(500)]
for K in (32, 16, 8):
    finals = np.array([[elo(df, k_factor=K, order=o)[m] for m in MODELS] for o in orders])
    sd = finals.std(0, ddof=1); first = (finals.argmax(1) == 0).mean(); mad = np.abs(finals - bt_elo).mean()
    print(f"K = {K}: SD across orders {np.round(sd, 1)}, M1 first {first:.2f}, mean |Elo - BT| {mad:.1f}, mean M6 rating {finals[:, 5].mean():.0f}")
