"""
A small check that the experiment tracking actually worked, not just
that the training script ran without crashing.

Run (after scripts/run_experiments.py has been run at least once):
    pytest -v

This does not retrain anything. It only reads back what MLflow
already recorded and checks the diary was kept properly: the right
number of runs, each with the metrics and params we expect, and at
least one run that beat the plain baseline on some real dimension.
"""

from pathlib import Path

import mlflow
import pytest

ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB_PATH = ROOT / "mlflow.db"
EXPERIMENT_NAME = "loan-default-classification"


@pytest.fixture(scope="module")
def runs_df():
    if not MLFLOW_DB_PATH.exists():
        pytest.skip("mlflow.db not found — run `python scripts/run_experiments.py` first.")
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None, f"Experiment '{EXPERIMENT_NAME}' not found in the tracking store."
    df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    return df


def test_four_runs_were_logged(runs_df):
    assert len(runs_df) == 4


def test_every_run_has_the_core_metrics(runs_df):
    required = [
        "metrics.roc_auc",
        "metrics.pr_auc",
        "metrics.precision_at_0.5",
        "metrics.recall_at_0.5",
        "metrics.f1_at_0.5",
        "metrics.train_seconds",
    ]
    for col in required:
        assert col in runs_df.columns, f"Missing metric column: {col}"
        assert runs_df[col].notna().all(), f"Some runs are missing {col}"


def test_every_run_has_a_model_type_param(runs_df):
    assert "params.model_type" in runs_df.columns
    assert runs_df["params.model_type"].notna().all()


def test_run_names_match_the_expected_sequence(runs_df):
    expected = {
        "01_logistic_regression_baseline",
        "02_tuned_decision_tree",
        "03_lightgbm",
        "04_lightgbm_smote_balanced",
    }
    actual = set(runs_df["tags.mlflow.runName"])
    assert actual == expected


def test_smote_run_improves_recall_over_plain_lightgbm(runs_df):
    """This is the actual experimental question Day 13 asked: does balancing
    the classes with SMOTE help the model catch more real defaulters?
    It should raise recall, even if it costs some precision or ROC AUC."""
    by_name = runs_df.set_index("tags.mlflow.runName")
    plain = by_name.loc["03_lightgbm", "metrics.recall_at_0.5"]
    smote = by_name.loc["04_lightgbm_smote_balanced", "metrics.recall_at_0.5"]
    assert smote > plain
