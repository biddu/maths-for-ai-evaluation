"""Generate arena300.csv: 300 head-to-head battles among six models M1..M6, judged by
judge J, which has a first-position bias.

True strengths on the Bradley-Terry (log) scale, centred to sum to zero:
    M1 clearly best, M2..M5 bunched within 0.15, M6 clearly worst.
Outcomes follow Davidson's tie model with tie parameter NU, and the model shown
first gets a bonus of POSITION on the log scale (the judge's bias). Pairs are
drawn uniformly at random and the presentation order is randomised.

Columns: battle, first, second, outcome   (outcome in {first, second, tie})
"""
import csv
import pathlib
import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "arena300.csv"
MODELS = [f"M{i}" for i in range(1, 7)]
BETA = np.array([1.00, 0.45, 0.40, 0.35, 0.30, -0.60])
BETA = BETA - BETA.mean()
NU = 0.25          # Davidson tie parameter
POSITION = 0.40    # judge's first-position bonus (log scale)
N_BATTLES = 300


def main(seed=111):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(N_BATTLES):
        i, j = rng.choice(6, 2, replace=False)
        pi_i = np.exp(BETA[i] + POSITION); pi_j = np.exp(BETA[j])
        tie = NU * np.sqrt(pi_i * pi_j)
        probs = np.array([pi_i, pi_j, tie]) / (pi_i + pi_j + tie)
        outcome = ["first", "second", "tie"][rng.choice(3, p=probs)]
        rows.append((b, MODELS[i], MODELS[j], outcome))
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["battle", "first", "second", "outcome"]); w.writerows(rows)
    oc = [r[3] for r in rows]
    print(f"{N_BATTLES} battles: first wins {oc.count('first')}, second wins {oc.count('second')}, ties {oc.count('tie')}; "
          f"true beta {np.round(BETA, 3)} -> {OUT}")


if __name__ == "__main__":
    main()
