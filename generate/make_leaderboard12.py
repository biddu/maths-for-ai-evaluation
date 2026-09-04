"""Generate leaderboard12.csv: 12 models (M1..M12) on 6 benchmarks with per-item outcomes.

Structure of a real leaderboard: every model is scored on the same items, item
difficulty is shared across models (so paired comparisons have power), and the
models' true strengths are bunched at the top. Models M1..M4 are designed to be
close; M2 and M3 are exactly equal in true strength.

Columns: benchmark, item, M1, ..., M12   (0/1 each)
"""
import csv
import pathlib
import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "leaderboard12.csv"
SIZES = {"B1": 200, "B2": 250, "B3": 400, "B4": 500, "B5": 800, "B6": 1000}
# true model strengths on the logit scale (M2 == M3 by construction)
STRENGTH = [1.55, 1.45, 1.45, 1.40, 1.10, 1.00, 0.85, 0.70, 0.50, 0.30, 0.05, -0.20]
# benchmark difficulty offsets
OFFSET = {"B1": 0.3, "B2": -0.4, "B3": 0.0, "B4": 0.6, "B5": -0.2, "B6": 0.1}


def main(seed=29):
    rng = np.random.default_rng(seed)
    rows = []
    for b, n in SIZES.items():
        diff = rng.normal(0.0, 1.5, n)                        # shared item difficulty
        for i in range(n):
            out = []
            for s in STRENGTH:
                logit = s + OFFSET[b] - diff[i] + rng.normal(0, 0.4)
                p = 1 / (1 + np.exp(-logit))
                out.append(int(rng.random() < p))
            rows.append([b, i] + out)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "item"] + [f"M{j}" for j in range(1, 13)])
        w.writerows(rows)
    arr = np.array([r[2:] for r in rows])
    print("macro-average scores:", np.round(arr.mean(axis=0), 3), "->", OUT)


if __name__ == "__main__":
    main()
