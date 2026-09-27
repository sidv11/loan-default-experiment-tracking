# Solution Write-Up: Loan Default Classification

Written in the style of a Kaggle solution post: what the problem was, what I tried, what worked, and what I'd do differently. This covers the same project as `README.md`, but reads it as one continuous story instead of a reference doc.

## Problem

Predict whether a loan applicant will default, using eight applicant details available at the time of the loan request: age, annual income, years at current job, credit score, loan amount, debt-to-income ratio, number of late payments in the last year, and whether there's a cosigner.

This matters because a bank has to decide, in seconds, whether an application is safe to approve. Getting it wrong in either direction costs money: approve a bad loan and it may never be repaid; reject a good applicant and the bank loses a paying customer.

## Data

8,000 synthetic loan applicants (about 24.5 percent default rate), generated with planted rules so the "true" pattern is known and any model's ranking of feature importance can be checked against ground truth. Full generation code is in `scripts/run_experiments.py`.

I built the data myself rather than pulling a public dataset for two reasons: it let me directly test whether SHAP and permutation importance (Day 11) actually find the true drivers of default, and it meant every downstream day in this series (explainability, deployment, tracking) works off the same known ground truth instead of restarting from scratch.

## EDA

The main things worth flagging before modeling:

- **Class imbalance.** Roughly 1 in 4 applicants defaults, not 1 in 2. A model that just always predicts "no default" would already be 75 percent accurate while being useless, so accuracy alone was never going to be the metric that mattered.
- **A near-duplicate feature pair.** `annual_income` and a second column meant to represent monthly income reported separately (used in the Day 11 explainability notebook) are almost perfectly correlated. Left unchecked, this splits importance between two columns that are really saying one thing.
- **No missing values, no extreme outliers** — this being synthetic data, real-world messiness (typos, unit mismatches, missing fields) isn't present, which is a real limitation noted below.

## Feature engineering

Kept deliberately minimal for this stage: the eight raw fields, no derived ratios beyond what's naturally present (debt-to-income was already given, not engineered). The earlier explainability work (Day 11) showed the model correctly leans on debt-to-income, loan amount, and credit score without needing hand-built interaction terms, so I didn't add any for this comparison — adding complexity that doesn't earn its keep just makes the model harder to explain later.

## Models tried (leaderboard)

Four approaches, each trained on the same 6,000/2,000 train/test split, tracked with MLflow (`scripts/run_experiments.py`, full detail in `README.md`):

| Rank (by F1) | Model | ROC AUC | PR AUC | Precision | Recall | F1 | Train time |
|---|---|---|---|---|---|---|---|
| 1 | LightGBM + SMOTE | 0.775 | 0.540 | 0.507 | **0.556** | **0.530** | 0.223s |
| 2 | Logistic regression | **0.802** | **0.571** | **0.626** | 0.399 | 0.488 | 0.010s |
| 3 | Decision tree | 0.739 | 0.475 | 0.518 | 0.462 | 0.489 | 0.014s |
| 4 | LightGBM (no balancing) | 0.783 | 0.553 | 0.605 | 0.393 | 0.477 | 0.165s |

## What worked

**Balancing the training data with SMOTE was the single biggest lever.** It's the only change between rows 1 and 4 in the table, and it moved recall from 0.393 to 0.556 — the model went from catching about 4 in 10 real defaulters to catching more than half. For a bank, missing an actual defaulter is usually the costlier mistake compared to being slightly too cautious with a safe applicant, so this is the version I'd actually ship if recall is the priority.

**Plain logistic regression was a stronger baseline than expected.** It has the best ROC AUC and PR AUC of all four models, beating LightGBM outright. This isn't a fluke: the rule used to generate defaults in this data is a weighted linear combination of the features, which is exactly the shape logistic regression is built to fit. It's also 15-20x faster to train than LightGBM.

## What didn't work, or didn't matter

**A single decision tree wasn't worth it.** Worse ROC AUC than every other model, and no real advantage in speed or interpretability over the simpler logistic regression. It sits in the comparison mainly to show the progression, not because it's a candidate to ship.

**LightGBM without balancing underperformed its own SMOTE-balanced version on every metric except precision.** More model complexity, in this case, bought nothing — the imbalance was the actual bottleneck, not the model family.

## Lessons learned

1. **A fancier model is not automatically a better one.** It depends on whether the real pattern needs the extra flexibility. Here it didn't, and logistic regression proved it.
2. **Fixing class imbalance mattered more than switching model families.** SMOTE alone closed most of the gap that model choice was expected to close.
3. **The metric you optimize for changes which model "wins."** By ROC AUC, logistic regression is best. By F1 and recall, SMOTE-balanced LightGBM is best. Neither answer is wrong; they're answering different questions about what the model is for.
4. **Correlated features can quietly hide their own importance** (seen in the income-pair test in Day 11's explainability notebook) — worth checking before trusting any single feature-importance ranking.

## What I'd try next

- Cross-validate across several splits instead of one, to see how stable this ranking really is.
- Try class weighting as an alternative to SMOTE, to see if it gets similar recall gains without synthesizing new rows.
- Tune LightGBM's hyperparameters properly instead of using one reasonable-looking default set — it's possible a tuned LightGBM would beat logistic regression even on this linear-shaped data.
- Test all of this on a real public loan dataset, since synthetic data this clean will always flatter simpler models a bit.

## Where the rest of the project lives

- Full run script and MLflow tracking setup: `scripts/run_experiments.py`, `README.md`
- Feature importance and per-customer explanations for the deployed model: loan default model explainability(https://github.com/sidv11/loan-default-model-explainability) notebook
- The trained model served as an API with a working frontend: Day 12's FastAPI + Streamlit project
