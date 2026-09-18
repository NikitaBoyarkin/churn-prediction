"""Tests for the churn-prediction pipeline.

Run: uv run --with pandas --with numpy --with scikit-learn --with lightgbm --with matplotlib --with pytest pytest -q
"""
import pandas as pd

from src import config as C
from src.features import build_dataset
from src.model import run_pipeline


def test_dataset_shape_and_balance():
    X, y = build_dataset(C.SNAPSHOT_TEST)
    assert len(X) > 1000
    assert "user_id" in X.columns
    for col in C.CATEGORICAL:
        assert col in X.columns
    rate = y.mean()
    assert 0.05 < rate < 0.30, f"churn rate {rate:.3f} outside expected band"
    assert int(y.sum()) > 100  # enough churners to learn from


def test_features_have_no_missing_after_fill():
    X, _ = build_dataset(C.SNAPSHOT_TEST)
    numeric = X.drop(columns=C.CATEGORICAL + ["user_id"])
    assert not numeric.isna().any().any()


def test_recency_feature_is_correct():
    """Recency = days between snapshot and last active day, computed only from
    past activity (no leakage)."""
    from src.features import load_raw
    snap = pd.Timestamp(C.SNAPSHOT_TEST)
    X, _ = build_dataset(C.SNAPSHOT_TEST)
    users, activity = load_raw()
    past = activity[activity.active_date < snap]
    for uid in X["user_id"].head(50):
        last = past[past.user_id == uid]["active_date"].max()
        if pd.isna(last):
            continue
        expected = (snap - last).days
        row = X[X.user_id == uid].iloc[0]
        assert row["recency_days"] == expected, (uid, row["recency_days"], expected)
        return


def test_pipeline_beats_baseline():
    r = run_pipeline(plot=False)
    m = r["lgbm_test"]
    assert m["roc_auc"] > 0.75
    assert m["pr_auc"] > 0.50
    assert m["lift_at_top_decile"] > 2.0
    # LGBM must be better-calibrated than balanced logreg on this data
    assert m["brier"] < r["logreg_test"]["brier"]


def test_time_split_is_chronological():
    assert C.SNAPSHOT_TRAIN < C.SNAPSHOT_VAL < C.SNAPSHOT_TEST
