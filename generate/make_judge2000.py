"""Generate judge2000.csv: 2,000 responses from model K, each labelled acceptable
(1) or not (0) by two LLM judges, with a 200-item human calibration subset.

Ground truth (known here, so the chapter can check itself):
    true acceptance rate   theta = 0.50
    judge J                sensitivity 0.90, specificity 0.70   (the failure-box judge)
    judge L                sensitivity 0.96, specificity 0.90   (a judge worth correcting)

The calibration subset is a simple random sample of 200 items on which an
adjudicated human label is available; the human label is taken to be correct.
The column `truth` holds the true label for every item. No estimator in the
chapter uses it; it is there so the reader can check every estimate.

Columns: item, calib, human, J, L, truth
"""
import csv
import pathlib
import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "judge2000.csv"
N = 2000
N_CALIB = 200
THETA = 0.50
JUDGES = {"J": (0.90, 0.70), "L": (0.96, 0.90)}


def main(seed=80):
    rng = np.random.default_rng(seed)
    truth = (rng.random(N) < THETA).astype(int)
    calib = np.zeros(N, int)
    calib[rng.choice(N, N_CALIB, replace=False)] = 1
    cols = {}
    for name, (s, t) in JUDGES.items():
        u = rng.random(N)
        cols[name] = np.where(truth == 1, (u < s).astype(int), (u >= t).astype(int))
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "calib", "human", "J", "L", "truth"])
        for i in range(N):
            w.writerow([i, calib[i], truth[i] if calib[i] else "", cols["J"][i], cols["L"][i], truth[i]])
    print(f"{N} items, true rate {truth.mean():.3f}, J rate {cols['J'].mean():.3f}, L rate {cols['L'].mean():.3f}, "
          f"calibration {N_CALIB} (human rate {truth[calib == 1].mean():.3f}) -> {OUT}")


if __name__ == "__main__":
    main()
