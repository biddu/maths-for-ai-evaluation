"""Exercise 7.7: Krippendorff's alpha with missing data, and what missingness does."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, krippendorff
from worked_example import krippendorff_alpha, ROOT

d = pd.read_csv(ROOT / "data" / "raters300.csv")
q = d.pivot(index="item", columns="rater", values="quality")
cats = [1, 2, 3, 4, 5]
Q = q[["R1", "R2", "R3"]].to_numpy().T.astype(float)
for lvl in ("nominal", "ordinal", "interval"):
    mine = krippendorff_alpha(Q, lvl, cats)
    ref = krippendorff.alpha(reliability_data=Q, level_of_measurement=lvl, value_domain=cats)
    assert abs(mine - ref) < 1e-10
    print(f"{lvl:9s} alpha = {mine:.4f} (package {ref:.4f})")
a_full = krippendorff_alpha(Q, "interval", cats)
a_noR3 = krippendorff_alpha(Q[:2], "interval", cats)
complete = ~np.isnan(Q).any(axis=0)
a_drop = krippendorff_alpha(Q[:, complete], "interval", cats)
# delete the 24 items with the largest spread among raters (on complete items)
spread = np.nanmax(Q, axis=0) - np.nanmin(Q, axis=0)
Qc = Q[:, complete]
spread_c = spread[complete]
keep = np.argsort(-spread_c)[24:]
# compare like with like: complete items minus the 24 widest, against complete items
a_hard = krippendorff_alpha(Qc[:, np.sort(keep)], "interval", cats)
print(f"interval alpha: gaps in place {a_full:.3f}; R3 removed {a_noR3:.3f}; R3's missing items dropped for all {a_drop:.3f}; 24 widest-spread items dropped {a_hard:.3f}")
