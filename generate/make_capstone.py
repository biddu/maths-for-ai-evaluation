"""Generate the Chapter 12 capstone: five models M1..M5 evaluated on five benchmarks,
in the form a leaderboard-style results table is built from. M3 is the incumbent
(the baseline every other model is compared with).

Files written to data/:
  capstone_knowledge.csv   500 shared items, columns item, M1..M5 (0/1)
  capstone_reading.csv     40 passages x 8 questions, columns passage, item, M1..M5 (0/1)
  capstone_judged.csv      400 responses per model labelled by judge L (s 0.96, t 0.90),
                           with an adjudicated human label on a random 100 per model:
                           columns model, item, judge, human (blank if not labelled), truth
  capstone_code.csv        100 items x 10 samples per model, columns item, sample, M1..M5
  capstone_battles.csv     400 pairwise battles judged by J (first-position bias 0.4,
                           Davidson ties nu 0.25): columns battle, first, second, outcome

True effects (log-odds offsets relative to M3, or rates) are in the constants below.
"""
import csv
import pathlib
import numpy as np

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
MODELS = [f"M{i}" for i in range(1, 6)]
KNOW_OFF = [0.35, 0.15, 0.00, -0.10, -0.45]
READ_OFF = [0.30, 0.10, 0.00, 0.05, -0.30]
JUDGE_THETA = [0.62, 0.58, 0.55, 0.50, 0.45]
CODE_OFF = [0.50, 0.20, 0.00, 0.10, -0.40]
BT_BETA = np.array([0.50, 0.20, 0.00, 0.10, -0.60]); BT_BETA = BT_BETA - BT_BETA.mean()
S_L, T_L = 0.96, 0.90
POSITION, NU = 0.40, 0.25


def sig(x):
    return 1 / (1 + np.exp(-x))


def main(seed=121):
    rng = np.random.default_rng(seed)
    DATA.mkdir(exist_ok=True)
    # knowledge: shared difficulty, 500 items
    diff = rng.normal(0.9, 1.2, 500)
    rows = [[i] + [int(rng.random() < sig(diff[i] + o)) for o in KNOW_OFF] for i in range(500)]
    with (DATA / "capstone_knowledge.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["item"] + MODELS); w.writerows(rows)
    # reading: 40 passages x 8 questions, strong passage effect
    rows = []; q = 0
    for p in range(40):
        pe = rng.normal(0.8, 1.0)
        for _ in range(8):
            qe = pe + rng.normal(0, 0.5)
            rows.append([p, q] + [int(rng.random() < sig(qe + o)) for o in READ_OFF]); q += 1
    with (DATA / "capstone_reading.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["passage", "item"] + MODELS); w.writerows(rows)
    # judged: 400 responses per model, judge L, human on 100
    rows = []
    for m, th in zip(MODELS, JUDGE_THETA):
        truth = (rng.random(400) < th).astype(int)
        u = rng.random(400)
        judge = np.where(truth == 1, (u < S_L).astype(int), (u >= T_L).astype(int))
        human_idx = set(rng.choice(400, 100, replace=False))
        for i in range(400):
            rows.append([m, i, int(judge[i]), int(truth[i]) if i in human_idx else "", int(truth[i])])
    with (DATA / "capstone_judged.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "item", "judge", "human", "truth"]); w.writerows(rows)
    # code: 100 items x 10 samples, logit-normal p_i with shared difficulty
    cdiff = rng.normal(0.6, 1.5, 100)
    rows = []
    for i in range(100):
        ps = [sig(cdiff[i] + o) for o in CODE_OFF]
        for s in range(10):
            rows.append([i, s] + [int(rng.random() < p) for p in ps])
    with (DATA / "capstone_code.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["item", "sample"] + MODELS); w.writerows(rows)
    # battles: 400 uniform pairs, judge J with position bias, Davidson ties
    rows = []
    for b in range(400):
        i, j = rng.choice(5, 2, replace=False)
        pi_i, pi_j = np.exp(BT_BETA[i] + POSITION), np.exp(BT_BETA[j])
        tie = NU * np.sqrt(pi_i * pi_j)
        out = ["first", "second", "tie"][rng.choice(3, p=np.array([pi_i, pi_j, tie]) / (pi_i + pi_j + tie))]
        rows.append([b, MODELS[i], MODELS[j], out])
    with (DATA / "capstone_battles.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["battle", "first", "second", "outcome"]); w.writerows(rows)
    print("capstone files written to", DATA)


if __name__ == "__main__":
    main()
