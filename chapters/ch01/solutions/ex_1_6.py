import numpy as np
rng = np.random.default_rng(0)
p, reps = 0.7, 200_000
for name, n in (("HumanEval", 164), ("GSM8K", 1319), ("MMLU", 14042)):
    gap = np.abs(rng.binomial(n, p, reps) - rng.binomial(n, p, reps)) / n
    print(name, (gap > 0.01).mean(), np.quantile(gap, 0.95) * 100)
