"""Generate raters300.csv: three human raters (R1, R2, R3) label 300 model responses.

Each response has a latent true state: a binary policy flag (prevalence about
8%) and a latent quality on a continuous scale. Each rater reports

    flag     0/1, with rater-specific sensitivity and specificity
    quality  1..5, an ordinal cut of latent quality plus rater noise and a
             rater-specific severity shift (R3 is harsher by half a point)

R3 did not rate quality on 24 of the items (missing, coded as empty), which
is the situation Krippendorff's alpha handles and Fleiss' kappa does not.

Raters are people, not models, so they do not join the model cast and are
named R1, R2, R3 (the naming convention is described in frontmatter/cast.tex).

Columns: item, rater, flag, quality
"""
import csv
import pathlib
import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "raters300.csv"
N = 300
PREVALENCE = 0.08
RATERS = ["R1", "R2", "R3"]
SENS = {"R1": 0.80, "R2": 0.75, "R3": 0.70}       # P(flag=1 | truly flagged)
SPEC = {"R1": 0.97, "R2": 0.96, "R3": 0.95}       # P(flag=0 | truly not flagged)
SEVERITY = {"R1": 0.0, "R2": 0.1, "R3": -0.5}     # shift on the latent quality scale
NOISE = {"R1": 0.55, "R2": 0.60, "R3": 0.65}      # rater noise SD on latent scale
CUTS = [-1.2, -0.4, 0.4, 1.2]                     # latent -> 1..5
N_MISSING_R3 = 24


def main(seed=71):
    rng = np.random.default_rng(seed)
    true_flag = rng.random(N) < PREVALENCE
    latent = rng.normal(0.0, 1.0, N)
    missing = set(rng.choice(N, N_MISSING_R3, replace=False))
    rows = []
    for i in range(N):
        for r in RATERS:
            if true_flag[i]:
                f = int(rng.random() < SENS[r])
            else:
                f = int(rng.random() >= SPEC[r])
            if r == "R3" and i in missing:
                q = ""
            else:
                z = latent[i] + SEVERITY[r] + rng.normal(0.0, NOISE[r])
                q = int(1 + np.searchsorted(CUTS, z))
            rows.append((i, r, f, q))
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "rater", "flag", "quality"])
        w.writerows(rows)
    print(f"{N} items x {len(RATERS)} raters, true prevalence {true_flag.mean():.3f} -> {OUT}")


if __name__ == "__main__":
    main()
