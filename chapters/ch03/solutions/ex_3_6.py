import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from worked_example import mcnemar_exact, mcnemar_chi2
for m in range(8, 400, 2):
    b, c = (m + 6) // 2, (m - 6) // 2
    if abs(mcnemar_exact(b, c) - mcnemar_chi2(b, c, True)[1]) < 0.001:
        print(m); break          # 18
