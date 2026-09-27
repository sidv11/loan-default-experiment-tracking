# Day 13: MLOps Basics, Experiment Tracking with MLflow

This redoes the Day 8 classification progression (logistic regression to LightGBM) on the loan default model, but this time every attempt is written down properly with MLflow instead of just remembered.

## Explained like you are five

Imagine you are baking the same cake four times. Each time you change one thing: a different recipe, a different oven setting, adding one new ingredient. If you do not write anything down, by cake four you will have forgotten which change actually made it taste better.

MLflow is the notebook you write in every single time you bake. Before baking, you write down what you are about to try (the params: oven temperature, sugar amount). After baking, you write down how it turned out (the metrics: did people like it, how long did it take). Nothing gets erased. Cake one's notes stay right there next to cake four's notes, so you can flip back and compare instead of trusting your memory.

## Problem statement

Train four versions of a loan default classifier, each a genuine attempt to improve on the last, and track every run's settings and results with MLflow instead of only keeping the final model.

## Dataset

Same synthetic loan applicant data used in Day 11 and Day 12 (8,000 rows, about 24.5 percent default rate), regenerated inside `scripts/run_experiments.py` so the whole thing runs from nothing but this repo. Saved to `data/synthetic_loan_applicants.csv`.

## Approach

Four tracked runs, each logged as its own MLflow run under the experiment `loan-default-classification`:

1. **Logistic regression baseline** — the simplest model, the number every later run has to beat.
2. **Tuned decision tree** — a single tree, max depth 5, more flexible than a straight line.
3. **LightGBM** — an ensemble of many small trees voting together, same setup as Day 8/11/12.
4. **LightGBM plus SMOTE** — same LightGBM, but the training data is rebalanced first with SMOTE (synthetic minority oversampling), since defaulters are the minority class (about a quarter of applicants). This tests a real question: does balancing the classes actually help, or just move the problem around?

For every run, MLflow logs: model type and hyperparameters (params), and ROC AUC, PR AUC, precision, recall, F1, and training time (metrics). All of this lives in one local SQLite file, `mlflow.db`, created next to the script. No server, no account, no cloud needed.

Note on tracking backend: MLflow's plain-folder file store is in maintenance mode as of MLflow 3.x, so this uses `sqlite:///mlflow.db` instead, which is what MLflow itself now recommends for local tracking.

## Results

| Run | Model | ROC AUC | PR AUC | Precision @0.5 | Recall @0.5 | F1 @0.5 | Train time (s) |
|---|---|---|---|---|---|---|---|
| 01 logistic regression baseline | logistic_regression | 0.802 | 0.571 | 0.626 | 0.399 | 0.488 | 0.010 |
| 02 tuned decision tree | decision_tree | 0.739 | 0.475 | 0.518 | 0.462 | 0.489 | 0.014 |
| 03 lightgbm | lightgbm | 0.783 | 0.553 | 0.605 | 0.393 | 0.477 | 0.165 |
| 04 lightgbm + SMOTE | lightgbm | 0.775 | 0.540 | 0.507 | 0.556 | **0.530** | 0.223 |

Chart: `images/run_comparison.png`. Full table: `data/run_comparison.csv`.

**The honest, slightly surprising finding:** LightGBM does not win on ROC AUC here. Plain logistic regression scores highest (0.802). This makes sense once you remember the underlying rule in the data is a simple weighted add-up of the features (the same kind of rule Day 11 planted on purpose), which is exactly what logistic regression is built for. A fancier model is not automatically a better one; it depends on whether the real pattern needs the extra flexibility.

**Where SMOTE actually helped:** balancing the classes did not improve ROC AUC, but it raised recall from 0.393 to 0.556 and F1 from 0.477 to 0.530, the best F1 of all four runs. In a loan default problem, missing an actual defaulter is usually the more expensive mistake, so depending on the business goal, run 4 could be the one worth shipping even though it does not have the top ROC AUC.

Automated check: `pytest tests/` (5 of 5 pass) confirms all four runs were logged with the right params and metrics, and specifically checks that the SMOTE run really did raise recall over the plain LightGBM run, not just that the script ran without crashing.

## Try it yourself

```bash
pip install -r requirements.txt
python scripts/run_experiments.py
```

This regenerates the data, trains all four models, logs everything to `mlflow.db`, and saves the comparison table and chart.

To browse the tracked runs in a web UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://127.0.0.1:5000` and look under the `loan-default-classification` experiment. Each run shows its params and metrics, and MLflow's own compare view can plot them against each other.

To confirm the tracking worked correctly without opening the UI:

```bash
pytest tests/ -v
```

## Why this matters

- **A comparison table beats a single final number.** "My model gets 0.53 F1" means little on its own. "Here are four honest attempts, and here is why I picked the fourth one" is what shows real engineering judgment.
- **Nothing gets lost.** Without tracking, trying a fifth idea next week means either re-running all four old models to remember their numbers, or trusting memory, which is exactly how "I think LightGBM was better" claims go unchecked.
- **This scales.** Four runs fit in your head. Forty do not. The habit of logging every attempt is the same at 4 runs or 400; only the tooling around it (a real server, a team dashboard) grows.
- **It caught a real result, not just a demo.** The SMOTE experiment answered a genuine question with numbers instead of a guess, and the answer (yes for recall and F1, no for ROC AUC) is the kind of nuance a hiring manager wants to see you notice.

## Honest limits

- Only one train/test split. A more careful version would repeat each run over several splits or use cross-validation and log the spread, not just one number.
- No hyperparameter search; the decision tree and LightGBM settings are reasonable defaults, not tuned.
- `mlflow.db` is a single local file with no team access, no remote artifact store, and no model registry.
- SMOTE was only tried on top of LightGBM, not on the baseline models, so we cannot fully separate "SMOTE helps" from "SMOTE helps LightGBM specifically."

## What I would improve next

1. Log a small hyperparameter sweep (for example, LightGBM's `num_leaves` and `learning_rate`) as additional MLflow runs, and use MLflow's parallel-coordinates view to see which settings matter.
2. Add cross-validation and log the mean and standard deviation of each metric, not a single train/test split.
3. Register the best run's model in the MLflow Model Registry, so it can be pulled by version into the Day 12 FastAPI service instead of a separate `joblib` file.
4. Try SMOTE (and a class-weight alternative) on the baseline models too, to isolate its effect from the model choice.
5. Point `mlflow.set_tracking_uri` at a shared server when working with a team, instead of a local SQLite file.

## Repository layout

```
day-13-mlops-experiment-tracking/
  README.md
  requirements.txt
  .gitignore
  mlflow.db                          SQLite tracking store, all 4 runs
  scripts/
    run_experiments.py                Generates data, trains + logs all 4 runs, saves chart
  tests/
    test_experiment_log.py             Confirms the tracking store recorded runs correctly
  data/
    synthetic_loan_applicants.csv
    run_comparison.csv
  images/
    run_comparison.png
```
