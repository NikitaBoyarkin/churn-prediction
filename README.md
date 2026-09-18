# Churn Prediction

A leakage-free churn model for a subscription product. The point of this
project is the **discipline**, not the algorithm: features are computed "as-of"
a snapshot date, the label is a future 30-day inactivity window, and the split
is chronological (train → val → test snapshots) so the model is evaluated the
way it would be used in production.

## What it does

- **Time-aware features** computed at a snapshot: recency, activity in prior
  7/14/30d windows, tenure, recent trend, average sessions, plus categorical
  channel/device/country/plan.
- **Leakage-free label**: churn = no activity in `[snapshot, snapshot+30d]` for
  users who were active in the prior 30d (i.e. real at-risk customers).
- **Chronological split**: train (2024-01-15) → val (2024-02-15) → test
  (2024-03-15). No random split — that would leak future activity into training.
- **Business metric first**: ROC-AUC and PR-AUC, but the decision metric is
  **recall@top-decile** and **lift@top-decile** — if retention acts on the top
  10% riskiest users, how many actual churners do we catch?
- **Native SHAP** via LightGBM's `predict_proba(pred_contrib=True)` — TreeSHAP
  without the `shap` package. `recency_days` dominates, as expected.
- **Baseline**: balanced logistic regression for comparison.

## Results (test snapshot, seed = 42)

| Metric | LightGBM | LogReg (balanced) |
|---|---|---|
| ROC-AUC | 0.904 | 0.917 |
| PR-AUC | 0.809 | 0.825 |
| Recall@top-decile | 0.53 | 0.54 |
| Lift@top-decile | 3.07× | 3.08× |
| **Brier (calibration)** | **0.068** | 0.099 |

AUC is a near-tie (the synthetic features are nearly linear, so logistic
regression is competitive) — but LightGBM is meaningfully better-calibrated
(Brier 0.068 vs 0.099), which matters when scores drive retention spend.
Honest finding: a strong recency feature makes the lift modest; the value of
the project is the leakage-free setup and the business metric, not a GBM trophy.

Top SHAP features: `recency_days` ≫ `channel` > `recent_trend` >
`avg_sessions_active_day` > `recency_ratio`.

## Data

Synthetic, deterministic (seed = 42). 12,000 users sign up over Jun 2023 – Jan
2024; each user has a channel-driven tenure (exponential lifetime) and a
plan-driven daily activity probability. Activity decays slightly approaching
churn. ~455k active user-days, churn rate ~16% per snapshot.

- `data/users.csv` — attributes + `death_date` (last possible active day).
- `data/activity.csv` — one row per active user-day (`sessions`, `events`).
- `data/generate_data.py` — the generator.

## Quick start

```bash
# Python >=3.10. Deps: pandas, numpy, scikit-learn, lightgbm, matplotlib.
uv run --with pandas --with numpy python data/generate_data.py   # build data
uv run --with pandas --with numpy --with scikit-learn --with lightgbm --with matplotlib python run.py
uv run --with pandas --with numpy --with scikit-learn --with lightgbm --with matplotlib --with pytest pytest -q
```

Outputs land in `reports/`: `metrics.json` + `evaluation.png` (ROC, PR, SHAP bar).

## Layout

```
churn-prediction/
├── data/
│   └── generate_data.py      # synthetic users + activity, seed=42
├── src/
│   ├── config.py             # snapshots, label window, model params
│   ├── features.py           # as-of feature builder + churn label
│   └── model.py              # train/eval/SHAP pipeline
├── tests/                    # balance, leakage, recency-correctness, beats-baseline
├── reports/                  # metrics.json + evaluation.png (gitignored)
├── conftest.py               # makes src importable
└── run.py
```

## Notes

- No random train/test split. A random split here would put a user's future
  activity in training and their past in test — silent leakage. The snapshot
  split is the whole reason this project exists.
- The label deliberately counts a user as churned only if they were active
  recently (prior 30d) and then go silent for 30d. Long-dead users are excluded,
  not labeled — predicting on ghosts is not a real task.
- LightGBM's native TreeSHAP avoids the `shap` package's `llvmlite` dependency,
  which has no wheel for Python 3.14. Same SHAP values, zero extra deps.
