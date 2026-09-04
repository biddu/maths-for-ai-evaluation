"""Exercise 8.8: position bias, three verdicts, agreement with the human."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import binom
from worked_example import ROOT

def position_bias(EF, FE):
    b = int(((EF == 1) & (FE == 0)).sum()); c = int(((EF == 0) & (FE == 1)).sum())
    p = min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5))
    M = EF.size; bias = (b - c) / M
    return b, c, p, bias, np.sqrt((b + c) / M - bias ** 2) / np.sqrt(M)

p = pd.read_csv(ROOT / "data" / "judge_pairs.csv")
EF, FE, REP = p.EF.to_numpy(), p.FE.to_numpy(), p.EF_rep.to_numpy()
print("swap:  b, c, p, bias, se =", position_bias(EF, FE))
print("repeat: b, c, p, bias, se =", position_bias(EF, REP))
win = (EF == 1) & (FE == 1); loss = (EF == 0) & (FE == 0); tie = ~win & ~loss
print(f"E wins: E-first {EF.mean():.3f}, F-first {FE.mean():.3f}, position-free win {win.mean():.3f} tie {tie.mean():.3f} loss {loss.mean():.3f}")
h = p.dropna(subset=["human"]); hy = h.human.astype(int).to_numpy()
hEF, hFE = h.EF.to_numpy(), h.FE.to_numpy()
hwin = (hEF == 1) & (hFE == 1); hloss = (hEF == 0) & (hFE == 0); decided = hwin | hloss
print(f"agreement with human: E-first {(hEF == hy).mean():.3f}, F-first {(hFE == hy).mean():.3f}, "
      f"position-free on {decided.sum()} decided prompts {(hwin[decided] == (hy[decided] == 1)).mean():.3f}")
