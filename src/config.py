"""Project configuration: snapshot dates, label window, columns, params."""
from __future__ import annotations

# Time-aware split: features computed "as of" snapshot; label = churn in next N days.
SNAPSHOT_TRAIN = "2024-01-15"
SNAPSHOT_VAL   = "2024-02-15"
SNAPSHOT_TEST  = "2024-03-15"
LABEL_WINDOW_DAYS = 30

# A user is a "customer" at snapshot if they had >=1 active day in the prior 30d.
PRIOR_ACTIVITY_DAYS = 30

RANDOM_SEED = 42

CATEGORICAL = ["channel", "device", "country", "plan"]

MODEL_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 30,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 400,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}
