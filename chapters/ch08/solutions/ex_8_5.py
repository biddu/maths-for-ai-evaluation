"""Exercise 8.5: budget split for judge L."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from worked_example import budget_split, observed_rate
theta, s, t = 0.5, 0.96, 0.90
J = s + t - 1; q = observed_rate(theta, s, t)
A = q * (1 - q) / J ** 2; B = (theta * s * (1 - s) + (1 - theta) * t * (1 - t)) / J ** 2
r = budget_split(theta, s, t, 20, 1)
budget = 5000
n_h = budget / (20 + 1 / r); N = n_h / r
print(f"J {J:.2f} q {q:.3f} A {A:.3f} B {B:.4f} ratio n_h/N {r:.3f}")
print(f"optimal: n_h {n_h:.0f}, N {N:.0f}, SE {math.sqrt(A / N + B / n_h):.4f}")
print(f"all humans: n_h {budget / 20:.0f}, SE {math.sqrt(0.25 / (budget / 20)):.4f}")
print(f"n_h 50, N 4000: SE {math.sqrt(A / 4000 + B / 50):.4f}")
