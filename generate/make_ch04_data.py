"""Generate the two datasets used in Chapter 4.

mtbench_judge.csv
    80 questions x 2 models (E, F), an integer 1-10 grade from a judge.
    Structure of MT-Bench single-answer grading: per-question difficulty
    shared by both models, scores bunched near the top with a ceiling at 10.
    Columns: question, model, score

mmlu_probs.csv
    2,000 multiple-choice items: the model's stated probability for its
    chosen answer and whether that answer was correct. The model is
    overconfident: stated probabilities exceed the true ones.
    Columns: item, confidence, correct

Both are synthetic with fixed seeds; the generating process is the point.
"""
import csv
import pathlib
import numpy as np

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def mtbench(seed=7):
    rng = np.random.default_rng(seed)
    q = 80
    difficulty = rng.normal(0, 1.1, q)                  # shared by both models
    latent_e = 7.9 - 1.2 * difficulty + rng.normal(0, 0.9, q)
    latent_f = 7.3 - 1.2 * difficulty + rng.normal(0, 0.9, q)
    def grade(x):
        return np.clip(np.rint(x), 1, 10).astype(int)
    e, f = grade(latent_e), grade(latent_f)
    with (DATA / "mtbench_judge.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["question", "model", "score"])
        for i in range(q):
            w.writerow([i, "E", int(e[i])])
            w.writerow([i, "F", int(f[i])])
    return e, f


def mmlu(seed=11):
    rng = np.random.default_rng(seed)
    n = 2000
    true_p = rng.beta(2.0, 1.3, n)                      # true probability the chosen answer is right
    true_p = 0.25 + 0.75 * true_p                       # chosen answer among four: at least chance
    # miscalibration on the logit scale: too sharp (slope > 1) and shifted upward,
    # so the model is overconfident when confident and slightly underconfident when not
    logit = np.log(true_p / (1 - true_p))
    conf = 1 / (1 + np.exp(-(1.8 * logit + 0.35)))
    conf = np.clip(conf + rng.normal(0, 0.03, n), 0.26, 0.995)
    correct = (rng.random(n) < true_p).astype(int)
    with (DATA / "mmlu_probs.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "confidence", "correct"])
        for i in range(n):
            w.writerow([i, f"{conf[i]:.4f}", int(correct[i])])
    return conf, correct


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    e, f = mtbench()
    conf, correct = mmlu()
    print(f"mtbench: E mean {e.mean():.2f}  F mean {f.mean():.2f}  E==10: {(e==10).mean():.2f}")
    print(f"mmlu: accuracy {correct.mean():.3f}  mean confidence {conf.mean():.3f}")
