"""Exercise 8.10: multi-class correction with a parametric bootstrap, and a confusable judge."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from worked_example import multiclass_correct

def run(M3, seed=90, N3=2000, n3=300, theta3=(0.5, 0.3, 0.2), B=4000):
    rng = np.random.default_rng(seed)
    truth = rng.choice(3, N3 + n3, p=theta3)
    judged = np.array([rng.choice(3, p=M3[y]) for y in truth])
    y_cal, f_cal, f_rest = truth[:n3], judged[:n3], judged[n3:]
    q_hat = np.bincount(f_rest, minlength=3) / N3
    M_hat = np.array([np.bincount(f_cal[y_cal == j], minlength=3) / (y_cal == j).sum() for j in range(3)])
    th = multiclass_correct(q_hat, M_hat)
    boots = np.empty((B, 3))
    for r in range(B):
        qb = rng.multinomial(N3, q_hat) / N3
        Mb = np.array([rng.multinomial(int((y_cal == j).sum()), M_hat[j]) / (y_cal == j).sum() for j in range(3)])
        boots[r] = multiclass_correct(qb, Mb)
    corr = np.corrcoef(boots.T)[1, 2]
    print(f"cond {np.linalg.cond(M3.T):6.2f}  theta_hat {np.round(th, 3)}  boot SE {np.round(boots.std(axis=0, ddof=1), 3)}  corr(2,3) {corr:.2f}")

M_book = np.array([[0.85, 0.12, 0.03], [0.20, 0.65, 0.15], [0.05, 0.25, 0.70]])
M_conf = np.array([[0.85, 0.12, 0.03], [0.20, 0.50, 0.30], [0.15, 0.45, 0.40]])
run(M_book); run(M_conf)
