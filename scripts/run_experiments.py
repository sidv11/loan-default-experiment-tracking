"""
Day 13: redo the Day 8 classification progression, but this time keep
a proper diary of every attempt using MLflow.

Explained simply: imagine you are baking the same cake four times,
each time changing one thing (more sugar, a different oven
temperature, a new pan). If you do not write anything down, by the
fourth cake you will have forgotten which change made it better or
worse. MLflow is the notebook you write in every time: what you
changed (params), how the cake turned out (metrics), and it keeps
every past attempt so you can compare them later instead of trusting
your memory.

This script trains four versions of a loan default classifier, each
one a genuine attempt to improve on the last, and logs every run to
a local MLflow tracking store (a SQLite file called mlflow.db,
created next to this script; no server or account needed).

Run:
    python scripts/run_experiments.py

Then look at the results either by running:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
and opening http://127.0.0.1:5000, or by reading the comparison
table and chart this script also saves to data/ and images/.
"""

import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB_PATH = ROOT / "mlflow.db"
DATA_DIR = ROOT / "data"
IMG_DIR = ROOT / "images"
EXPERIMENT_NAME = "loan-default-classification"


def make_data(n: int = 8000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Same planted-rule synthetic loan data used in Day 11 and Day 12, so this
    whole mini-series stays consistent about what the 'true' pattern is."""
    rng = np.random.default_rng(random_state)

    age = np.clip(rng.normal(38, 11, n), 21, 70)
    annual_income = np.clip(rng.lognormal(mean=np.log(55000), sigma=0.45, size=n), 15000, 300000)
    employment_years = np.clip(rng.uniform(0, 1, n) * (age - 20), 0, 40)
    credit_score = np.clip(rng.normal(660, 70, n) + 0.5 * employment_years, 300, 850)
    loan_to_income = rng.uniform(0.05, 0.8, n)
    loan_amount = annual_income * loan_to_income
    debt_to_income = np.clip(rng.normal(0.30, 0.10, n), 0.02, 0.8)
    late_rate = 0.5 + np.clip((700 - credit_score) / 100, 0, None) * 1.2
    num_late_payments = np.clip(rng.poisson(late_rate), 0, 12)
    has_cosigner = rng.binomial(1, 0.2, n)

    logit = (
        -2.6
        + 3.0 * loan_to_income
        + 4.0 * (debt_to_income - 0.30)
        - 0.012 * (credit_score - 660)
        + 0.35 * num_late_payments
        - 0.04 * employment_years
        - 0.8 * has_cosigner
    )
    prob_default = 1 / (1 + np.exp(-logit))
    defaulted = rng.binomial(1, prob_default)

    return pd.DataFrame(
        {
            "age": age.round(0),
            "annual_income": annual_income.round(0),
            "employment_years": employment_years.round(1),
            "credit_score": credit_score.round(0),
            "loan_amount": loan_amount.round(0),
            "debt_to_income": debt_to_income.round(3),
            "num_late_payments": num_late_payments,
            "has_cosigner": has_cosigner,
            "defaulted": defaulted,
        }
    )


def evaluate(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision_at_0.5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_0.5": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_0.5": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_experiment(name, params, train_fn, X_train, y_train, X_test, y_test):
    """One tracked attempt: train, time it, score it, log everything to MLflow."""
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        start = time.time()
        model = train_fn(X_train, y_train)
        train_seconds = time.time() - start

        proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, proba)
        metrics["train_seconds"] = round(train_seconds, 3)

        mlflow.log_metrics(metrics)
        mlflow.set_tag("stage", name)

        print(f"[{name}] " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
        return {"run": name, **params, **metrics}


def main():
    DATA_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)

    # MLflow's plain-folder "file store" is in maintenance mode as of MLflow 3.x,
    # so runs are tracked in a local SQLite database instead. This is still fully
    # local (one file, mlflow.db, next to this script) and needs no server or
    # account, but it is the backend MLflow itself now recommends.
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = make_data()
    df.to_csv(DATA_DIR / "synthetic_loan_applicants.csv", index=False)
    features = [c for c in df.columns if c != "defaulted"]
    X, y = df[features], df["defaulted"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Default rate: train={y_train.mean():.3f}, test={y_test.mean():.3f}\n")

    rows = []

    # Run 1: the simplest possible baseline. This is the number every later run must beat.
    def train_logreg(Xtr, ytr):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        m.fit(Xtr, ytr)
        return m

    rows.append(run_experiment(
        "01_logistic_regression_baseline",
        {"model_type": "logistic_regression", "max_iter": 1000},
        train_logreg, X_train, y_train, X_test, y_test,
    ))

    # Run 2: a single tuned decision tree. More flexible than a straight line, still simple.
    def train_tree(Xtr, ytr):
        from sklearn.tree import DecisionTreeClassifier
        m = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, random_state=RANDOM_STATE)
        m.fit(Xtr, ytr)
        return m

    rows.append(run_experiment(
        "02_tuned_decision_tree",
        {"model_type": "decision_tree", "max_depth": 5, "min_samples_leaf": 30},
        train_tree, X_train, y_train, X_test, y_test,
    ))

    # Run 3: LightGBM, an ensemble of many small trees voting together.
    lgb_params = dict(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1,
    )

    def train_lgbm(Xtr, ytr):
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(Xtr, ytr)
        return m

    rows.append(run_experiment(
        "03_lightgbm",
        {"model_type": "lightgbm", **{k: v for k, v in lgb_params.items() if k != "random_state"}},
        train_lgbm, X_train, y_train, X_test, y_test,
    ))

    # Run 4: same LightGBM, but balance the classes first with SMOTE (synthetic
    # minority oversampling). Defaults are the minority class here (about a
    # quarter of applicants), so this tests whether balancing actually helps.
    def train_lgbm_smote(Xtr, ytr):
        sm = SMOTE(random_state=RANDOM_STATE)
        Xtr_bal, ytr_bal = sm.fit_resample(Xtr, ytr)
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(Xtr_bal, ytr_bal)
        return m

    rows.append(run_experiment(
        "04_lightgbm_smote_balanced",
        {"model_type": "lightgbm", "balancing": "SMOTE", **{k: v for k, v in lgb_params.items() if k != "random_state"}},
        train_lgbm_smote, X_train, y_train, X_test, y_test,
    ))

    # --- Comparison table ---
    comparison = pd.DataFrame(rows)
    ordered_cols = ["run", "model_type", "roc_auc", "pr_auc", "precision_at_0.5", "recall_at_0.5", "f1_at_0.5", "train_seconds"]
    comparison_view = comparison[ordered_cols].round(4)
    comparison_view.to_csv(DATA_DIR / "run_comparison.csv", index=False)
    print("\nComparison table:\n", comparison_view.to_string(index=False))

    # --- Comparison chart ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    x = np.arange(len(comparison_view))
    short_labels = ["logreg\nbaseline", "decision\ntree", "lightgbm", "lightgbm\n+ SMOTE"]

    axes[0].bar(x, comparison_view["roc_auc"], color="#66c2a5")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_labels, fontsize=9)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title("ROC AUC by run")
    axes[0].set_ylabel("ROC AUC")
    for i, v in enumerate(comparison_view["roc_auc"]):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    axes[1].bar(x, comparison_view["f1_at_0.5"], color="#fc8d62")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_labels, fontsize=9)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("F1 at 0.5 threshold by run")
    axes[1].set_ylabel("F1")
    for i, v in enumerate(comparison_view["f1_at_0.5"]):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle("Model comparison across four tracked MLflow runs", y=1.02)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "run_comparison.png", bbox_inches="tight", dpi=130)
    print(f"\nSaved comparison chart to {IMG_DIR / 'run_comparison.png'}")
    print(f"Saved comparison table to {DATA_DIR / 'run_comparison.csv'}")
    print(f"\nAll runs logged under MLflow experiment '{EXPERIMENT_NAME}' in {MLFLOW_DB_PATH}")
    print("View them with: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
