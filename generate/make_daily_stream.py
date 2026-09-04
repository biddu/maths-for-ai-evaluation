"""Generate daily_stream.csv: a deployed model N evaluated on 200 fresh items every day
for 100 days. The true pass rate is 0.80 for days 1-39 and drops to 0.75 from day 40.

Chapter 10 uses this stream to compare stopping rules: a daily fixed-horizon test
(peeking), Wald's SPRT, a Beta-binomial confidence sequence and a CUSUM monitor.

Columns: day, item, pass
"""
import csv
import pathlib
import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "daily_stream.csv"
DAYS, PER_DAY = 100, 200
P_BASE, P_AFTER, CHANGE_DAY = 0.80, 0.75, 40


def main(seed=101):
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(1, DAYS + 1):
        p = P_BASE if day < CHANGE_DAY else P_AFTER
        outcomes = (rng.random(PER_DAY) < p).astype(int)
        rows.extend((day, i, int(o)) for i, o in enumerate(outcomes))
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "item", "pass"])
        w.writerows(rows)
    arr = np.array([r[2] for r in rows]).reshape(DAYS, PER_DAY)
    print(f"{DAYS} days x {PER_DAY} items; mean pass rate before day {CHANGE_DAY}: {arr[:CHANGE_DAY-1].mean():.3f}, "
          f"from day {CHANGE_DAY}: {arr[CHANGE_DAY-1:].mean():.3f} -> {OUT}")


if __name__ == "__main__":
    main()
