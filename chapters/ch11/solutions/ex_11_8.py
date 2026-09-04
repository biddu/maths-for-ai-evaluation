"""Exercise 11.8: greedy pair selection with two objectives (fewer replications than the chapter)."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from worked_example import bt_mm, bt_fisher_cov, se_diff, simulate_battle, K

def run(objective, n_total, rng, warm=60):
    W = np.zeros((K, K)); pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    contested = [1, 2, 3, 4]; chosen = np.zeros((K, K))
    for t in range(n_total):
        if t < warm:
            i, j = pairs[rng.integers(len(pairs))]
        else:
            beta, _, _ = bt_mm(W, iters=200, tol=1e-8); cov = bt_fisher_cov(beta, W)
            if objective == "top":
                order = np.argsort(-beta); targets = [(order[0], order[1])]
            else:
                targets = [(a, b) for a in contested for b in contested if a < b]
            best, best_val = None, -1
            for (a, b) in pairs:
                p = 1 / (1 + math.exp(beta[b] - beta[a])); v = p * (1 - p)
                d = np.zeros(K); d[a] = 1; d[b] = -1; cd = cov @ d; denom = 1 + v * (d @ cd)
                gain = max(v * ((np.eye(K)[c1] - np.eye(K)[c2]) @ cd) ** 2 / denom for c1, c2 in targets)
                if gain > best_val: best_val, best = gain, (a, b)
            i, j = best; chosen[i, j] += 1
        out = simulate_battle(i, j, rng)
        if out == 0: W[i, j] += 1
        elif out == 1: W[j, i] += 1
        else: W[i, j] += 0.5; W[j, i] += 0.5
    beta, _, _ = bt_mm(W); cov = bt_fisher_cov(beta, W)
    top = np.argsort(-beta)[:2]
    return (se_diff(cov, top[0], top[1]), max(se_diff(cov, a, b) for a in contested for b in contested if a < b), chosen)

rng = np.random.default_rng(118)
for objective in ("contested", "top"):
    res = [run(objective, 400, rng) for _ in range(8)]
    print(f"objective {objective}: SE(top gap) {np.mean([r[0] for r in res]):.3f}, SE(largest M2-M5 gap) {np.mean([r[1] for r in res]):.3f}")
    chosen = sum(r[2] for r in res) / len(res)
    print("   mean battles per pair chosen after warm-up (upper triangle):")
    print(np.round(chosen, 0))
