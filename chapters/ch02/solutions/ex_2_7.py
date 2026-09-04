import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from worked_example import wald, wilson, coverage

n, grid = 20, np.linspace(0.05, 0.95, 181)
rng = np.random.default_rng(0)
for name, m in (("wald", wald), ("wilson", wilson)):
    exact = np.array([coverage(m, n, p) for p in grid])
    sim = []
    for p in grid:
        s = rng.binomial(n, p, 100_000)
        los, his = np.array([m(k, n) for k in range(n + 1)]).T
        sim.append(np.mean((los[s] <= p) & (p <= his[s])))
    sim = np.array(sim)
    print(name, exact.min(), np.abs(exact - sim).max())
