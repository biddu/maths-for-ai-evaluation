"""Generate strata_subjects.csv: a subject-stratified benchmark for model I.

Eight subjects with a fixed 150 items each (1,200 items). Subject accuracies
differ widely, from about 0.45 to 0.92. Within a subject, items are
exchangeable. This is the design in which ignoring the structure OVERSTATES
the variance, because the between-subject spread is fixed by design and does
not contribute to sampling variance.

Columns: subject, item, correct
"""
import csv
import pathlib
import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "strata_subjects.csv"
SUBJECTS = ["algebra", "biology", "chemistry", "history", "law", "logic", "physics", "statistics"]
RATES = [0.62, 0.88, 0.71, 0.92, 0.45, 0.55, 0.66, 0.84]
PER_SUBJECT = 150


def main(seed=23):
    rng = np.random.default_rng(seed)
    rows = []
    q = 0
    for s, r in zip(SUBJECTS, RATES):
        for _ in range(PER_SUBJECT):
            rows.append((s, q, int(rng.random() < r)))
            q += 1
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "item", "correct"])
        w.writerows(rows)
    y = np.array([r[2] for r in rows])
    print(f"{len(SUBJECTS)} subjects x {PER_SUBJECT} items, accuracy {y.mean():.3f} -> {OUT}")


if __name__ == "__main__":
    main()
