"""Exercise 9.8: pass@k against budget for three cost ratios; the tie at a budget of 8."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from worked_example import pass_at_k, ROOT
d = pd.read_csv(ROOT / "data" / "humaneval_k20.csv")
counts = d.groupby(["model", "item"])["pass"].sum().unstack(0)
curve = {m: np.array([np.mean([pass_at_k(20, ci, k) for ci in counts[m]]) for k in range(1, 21)]) for m in "AB"}
def at_budget(m, budget, cost):
    k = int(budget // cost)
    return curve[m][min(k, 20) - 1] if k >= 1 else 0.0
for ratio in (1.0, 1.6, 3.0):
    print(f"cost ratio {ratio}: cost per solve A {ratio / curve['A'][0]:.2f}, B {1 / curve['B'][0]:.2f}")
    for budget in (1, 2, 4, 8, 12, 16, 20):
        print(f"   budget {budget:2d}: A {at_budget('A', budget, ratio):.3f}  B {at_budget('B', budget, 1.0):.3f}")
# tie at budget 8: scan ratios
for ratio in np.arange(1.0, 2.01, 0.05):
    a = at_budget("A", 8, ratio); b = at_budget("B", 8, 1.0)
    if a < b:
        print(f"first ratio at which B leads at budget 8: {ratio:.2f} (A {a:.3f} with k = {int(8 // ratio)}, B {b:.3f})"); break
