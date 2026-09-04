"""Generate judge_pairs.csv: judge J compares responses from models E and F on
400 prompts, in both presentation orders, with one order repeated.

Each prompt has a latent quality difference d = q_E - q_F ~ N(0.3, 1). The judge
sees the two responses in an order and prefers the FIRST one when
d_first + POSITION + VERBOSITY * (standardised length difference) + noise > 0.
POSITION = 0.5
response. The judge has a fixed idiosyncratic view of each prompt plus a small amount of
fresh noise on every call, so a repeat of the same order is highly consistent
while the swapped order is not. A human preference is available on a random 120 prompts
(no position or verbosity bias, own noise).

Columns:
    prompt, len_E, len_F,
    EF      1 if the judge preferred E when E was shown first, else 0
    FE      1 if the judge preferred E when F was shown first, else 0
    EF_rep  the E-first order judged a second time
    human   1 if the human preferred E, 0 if F, blank if not labelled
"""
import csv
import pathlib
import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "judge_pairs.csv"
N = 400
N_HUMAN = 120
POSITION = 0.5
VERBOSITY = 0.5
CALL_NOISE = 0.25      # fresh noise on every call
PROMPT_NOISE = 0.8     # the judge's fixed idiosyncrasy on each prompt (same on every call)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main(seed=81):
    rng = np.random.default_rng(seed)
    d = rng.normal(0.3, 1.0, N)                          # quality of E minus F
    len_E = np.round(np.exp(rng.normal(5.3, 0.4, N))).astype(int)
    len_F = np.round(np.exp(rng.normal(5.1, 0.4, N))).astype(int)
    zlen = (np.log(len_E) - np.log(len_F)) / 0.4        # standardised length difference

    e_prompt = rng.normal(0, PROMPT_NOISE, N)

    def judge(first_is_E, noise):
        # the judge prefers the FIRST response when its latent score is positive
        d_first = d if first_is_E else -d
        z_first = zlen if first_is_E else -zlen
        e_first = e_prompt if first_is_E else -e_prompt
        prefers_first = (d_first + POSITION + VERBOSITY * z_first + e_first + noise) > 0
        return np.where(first_is_E, prefers_first, ~prefers_first).astype(int)   # 1 = prefers E

    EF = judge(True, rng.normal(0, CALL_NOISE, N))
    EF_rep = judge(True, rng.normal(0, CALL_NOISE, N))
    FE = judge(False, rng.normal(0, CALL_NOISE, N))
    human_idx = set(rng.choice(N, N_HUMAN, replace=False))
    human_all = (rng.random(N) < sigmoid(d + rng.normal(0, 0.7, N))).astype(int)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "len_E", "len_F", "EF", "FE", "EF_rep", "human"])
        for i in range(N):
            w.writerow([i, len_E[i], len_F[i], EF[i], FE[i], EF_rep[i], human_all[i] if i in human_idx else ""])
    print(f"{N} prompts: P(E wins | E first) {EF.mean():.3f}, P(E wins | F first) {FE.mean():.3f}, "
          f"repeat consistency {(EF == EF_rep).mean():.3f}, order consistency {(EF == FE).mean():.3f} -> {OUT}")


if __name__ == "__main__":
    main()
