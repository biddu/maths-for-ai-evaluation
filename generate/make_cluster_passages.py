"""Generate cluster_passages.csv: reading-comprehension questions nested in passages.

60 passages, each with 3 to 8 questions (about 300 questions in all), and
model H's correctness on each question. A passage-level random effect makes
questions from the same passage correlated: a passage the model misreads
loses several questions at once.

Columns: passage, question, correct
"""
import csv
import pathlib
import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "cluster_passages.csv"


def main(seed=19):
    rng = np.random.default_rng(seed)
    n_pass = 60
    sizes = rng.integers(3, 9, n_pass)                      # 3..8 questions per passage
    passage_effect = rng.normal(0.0, 1.3, n_pass)           # between-passage spread on the logit scale
    rows = []
    q = 0
    for c in range(n_pass):
        for _ in range(sizes[c]):
            logit = 0.9 + passage_effect[c] + rng.normal(0, 0.6)
            p = 1 / (1 + np.exp(-logit))
            rows.append((c, q, int(rng.random() < p)))
            q += 1
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["passage", "question", "correct"])
        w.writerows(rows)
    y = np.array([r[2] for r in rows])
    print(f"{n_pass} passages, {len(rows)} questions, accuracy {y.mean():.3f}, sizes {sizes.min()}..{sizes.max()} -> {OUT}")


if __name__ == "__main__":
    main()
