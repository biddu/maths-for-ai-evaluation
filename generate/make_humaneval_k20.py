"""Generate the book's HumanEval-shaped dataset: 164 items x 2 models x 20 samples.

The dataset is synthetic but structured like real per-item benchmark output:
each item i has a latent per-item success probability p_i for each model, and
each of the 20 samples is a Bernoulli draw from it. Item difficulties are
correlated across the two models (hard problems are hard for both), which is
what makes the paired analysis of Chapter 3 matter.

The seed is chosen so that the first sample of each model reproduces the
running example of Chapter 1: model A scores 117/164, model B 115/164.

Output: data/humaneval_k20.csv with columns
    item, model, sample, pass  (pass in {0,1})
and data/humaneval_k20_truth.csv with the latent p_i of each item for each
model (columns item, p_A, p_B). No estimator in the book uses the truth file;
Chapter 9 uses it to check the Beta extrapolation of pass@k beyond n = 20.
"""
import csv
import pathlib
import numpy as np

N_ITEMS, K, TARGET_A, TARGET_B = 164, 20, 117, 115
OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "humaneval_k20.csv"
TRUTH = OUT.with_name("humaneval_k20_truth.csv")


def draw(seed):
    rng = np.random.default_rng(seed)
    # latent difficulty on the logit scale, shared component + model-specific
    shared = rng.normal(1.0, 1.6, N_ITEMS)
    logit_a = shared + rng.normal(0.15, 0.5, N_ITEMS)
    logit_b = shared + rng.normal(0.05, 0.5, N_ITEMS)
    p_a, p_b = 1 / (1 + np.exp(-logit_a)), 1 / (1 + np.exp(-logit_b))
    s_a = (rng.random((N_ITEMS, K)) < p_a[:, None]).astype(int)
    s_b = (rng.random((N_ITEMS, K)) < p_b[:, None]).astype(int)
    return s_a, s_b, p_a, p_b


def main():
    for seed in range(100_000):
        s_a, s_b, p_a, p_b = draw(seed)
        if s_a[:, 0].sum() == TARGET_A and s_b[:, 0].sum() == TARGET_B:
            break
    else:
        raise SystemExit("no seed found")
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "model", "sample", "pass"])
        for name, s in (("A", s_a), ("B", s_b)):
            for i in range(N_ITEMS):
                for j in range(K):
                    w.writerow([i, name, j, int(s[i, j])])
    with TRUTH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "p_A", "p_B"])
        for i in range(N_ITEMS):
            w.writerow([i, f"{p_a[i]:.6f}", f"{p_b[i]:.6f}"])
    print(f"seed={seed}  A={s_a[:,0].sum()}  B={s_b[:,0].sum()}  -> {OUT}")


if __name__ == "__main__":
    main()
