"""Generate judge_labels.csv: model J, an LLM judge, labels 600 responses as
acceptable (1) or not (0), alongside an adjudicated human label.

The human label is treated as the reference. Acceptable responses are common
(prevalence about 0.88). The judge is very good on acceptable responses
(sensitivity 0.95) and mediocre on unacceptable ones (specificity 0.60).
The result is an exact-match rate around 0.90 and a Cohen's kappa around
0.55: the "kappa deflation" gap of roughly 35 points that the chapter
discusses.

Columns: item, human, J
"""
import csv
import pathlib
import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "judge_labels.csv"
N = 600
PREVALENCE = 0.88
SENS = 0.95
SPEC = 0.60


def main(seed=77):
    rng = np.random.default_rng(seed)
    human = (rng.random(N) < PREVALENCE).astype(int)
    u = rng.random(N)
    judge = np.where(human == 1, (u < SENS).astype(int), (u >= SPEC).astype(int))
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "human", "J"])
        w.writerows(zip(range(N), human, judge))
    print(f"{N} items, prevalence {human.mean():.3f}, exact match {(human == judge).mean():.3f} -> {OUT}")


if __name__ == "__main__":
    main()
