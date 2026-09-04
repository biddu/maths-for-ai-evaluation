# Maths for AI Evaluation: companion code

Repository: <https://github.com/biddu/maths-for-AI-evolution>

Code and data for *Maths for AI Evaluation: Confidence Intervals, Judges, and Leaderboards* by Avishek Nag (Mathematics for Everything, Book 5).

Every number printed in the book is produced by a script in this repository, from data sets that the repository also generates. Nothing in the text was typed by hand. If you can run Python, you can check the book.

## What is here

    generate/                 scripts with fixed seeds that build the data sets in data/
    data/                     the data sets exactly as the book uses them (CSV)
    chapters/chNN/
        worked_example.py     regenerates every number printed in Chapter NN -> numbers.json
        numbers.json          the numbers as the book prints them
        figures.py            regenerates the chapter's figures (greyscale PDF)
        fig_*.pdf             the figures as printed
        solutions/ex_N_k.py   complete scripts for the coding exercises
    requirements.txt          Python packages
    run_all.sh, Makefile      rebuild everything from scratch

The data are synthetic. Model A of Chapter 1 is not a product and the 164 items it is scored on are not HumanEval, though they have its size and shape. Synthetic data let the book state the truth: when Chapter 8 corrects a judge's approval rate from 61% to 50%, `data/judge2000.csv` carries a `truth` column against which the correction can be checked, and `data/humaneval_k20_truth.csv` holds the latent per-item probabilities behind the pass@k curves of Chapter 9.

## Running it

Python 3.10 or later.

    pip install -r requirements.txt
    ./run_all.sh            # or: make

`run_all.sh` regenerates the data, then every chapter's `numbers.json` and figures, and finally runs the exercise solutions. It takes a few minutes; Chapters 3 and 5 run bootstrap simulations. To rebuild one chapter:

    python chapters/ch08/worked_example.py
    python chapters/ch08/figures.py

Each `worked_example.py` prints a readable summary and writes `numbers.json` beside itself. The scripts implement every method from scratch (a Rogan-Gladen correction is four lines, Cohen's kappa is six) and then, where a library implements the same quantity, assert that the two agree. The libraries are there as a check, not as the method.

Scripts must be run from the repository, not copied elsewhere: each one locates `data/` relative to its own path.

## Chapter guide

| Chapter | Topic | Data | Main functions in `worked_example.py` |
|---|---|---|---|
| 1 | A score is an estimate | humaneval_k20 | `se_binary` |
| 2 | Confidence intervals | humaneval_k20 | `wald`, `wilson`, `clopper_pearson`, `agresti_coull`, `jeffreys`, `coverage` |
| 3 | Comparing two models on shared items | gsm8k_pairs | `mcnemar_exact`, `paired_bootstrap`, `tango_interval`, `noninferiority`, `power_mcnemar` |
| 4 | Continuous and graded scores | mtbench_judge, mmlu_probs | `t_interval`, `bootstrap`, `bca_ci`, `signed_rank`, `brier_decomposition` |
| 5 | Clustered items | cluster_passages, strata_subjects | `balanced_components`, `se_cluster`, `cluster_bootstrap`, `optimal_k` |
| 6 | Leaderboards and multiple comparisons | leaderboard12 | `bonferroni`, `holm`, `benjamini_hochberg`, `expected_max_normal` |
| 7 | Rater agreement | raters300, judge_labels | `cohen_kappa`, `kappa_se`, `kappa_max`, `concordance_cc`, `fleiss_kappa`, `krippendorff_alpha`, `icc_two_way` |
| 8 | The judge as an instrument | judge2000, judge_pairs | `rogan_gladen`, `rg_variance`, `judge_worth_it`, `budget_split`, `calibration_size`, `multiclass_correct`, `ppi_mean`, `bayes_posterior`, `metropolis` |
| 9 | pass@k | humaneval_k20 (+truth) | `pass_at_k`, `pass_pow_k`, `beta_fit_counts`, `beta_pass_at_k`, `beta_majority`, `best_of_n_ceiling` |
| 10 | Sequential testing and monitoring | daily_stream, judge_pairs | `peeking_sim`, `sprt`, `beta_binomial_cs`, `cusum_path`, `cusum_arl0` |
| 11 | Pairwise preferences and Bradley-Terry | arena300 | `win_matrix`, `bt_mm`, `bt_fisher_cov`, `logistic_fit`, `elo`, `bootstrap_bt`, `run_schedule` |
| 12 | Reporting | capstone_* (five files) | the capstone table, naive and honest |

## The models

One letter is one model for the whole book and letters are never reused. A and B are the code models of Chapters 1, 3, 5 and 9; C and D the arithmetic models of Chapter 3; E and F the chat models of Chapters 4 and 10; G the multiple-choice model of Chapter 4; H and I the models of Chapter 5; J, K and L the judges and judged model of Chapters 7 and 8; N the monitored model of Chapter 10. Models that appear only as leaderboard entries are numbered M1, M2, ... within their chapter, and human raters are R1, R2, R3.

## Reproducibility

Every generator sets a NumPy seed and the data files in `data/` are what those seeds produce. Running `generate/` again overwrites them with byte-identical files; running `worked_example.py` again reproduces `numbers.json` up to floating-point noise in the last printed digit. Package versions used for the printed edition are listed in `requirements.txt`.

## Licence

The code is released under the MIT Licence (see `LICENSE`). The data are synthetic and carry the same licence. The book's text is copyright Avishek Nag and is not part of this repository.

## Citing

    Avishek Nag, Maths for AI Evaluation: Confidence Intervals, Judges, and Leaderboards.
    Mathematics for Everything, Book 5, 2027.

A `CITATION.cff` file is included for reference managers.
