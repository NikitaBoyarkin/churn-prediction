# Churn Prediction

> Leakage-free churn model on 12,000 users: test lift@top-decile **5.44×** (recall 0.544), at-risk AUC 0.901 — and an honest tie with LogReg.

## Navigation
- 🌐 [Portfolio site](https://nikitaboyarkin.github.io)
- 🏦 [Flagship case: Volta neobank](https://github.com/NikitaBoyarkin/volta-banking)
- 🧪 Related: [rfm-analysis-of-bank-clients](https://github.com/NikitaBoyarkin/rfm-analysis-of-bank-clients) · [sql-analytics-case-study](https://github.com/NikitaBoyarkin/sql-analytics-case-study) · [tableau_cohort_analysis](https://github.com/NikitaBoyarkin/tableau_cohort_analysis)

**▶ [Diagram: Leakage-free churn pipeline](docs/diagrams/churn-prediction.html)** — opens in a browser, no install, no build

**On this page:** [Situation](#situation) · [Task](#task) · [Actions](#actions) · [Result](#result) · [What it does](#what-it-does) · [At-risk subgroup](#at-risk-subgroup) · [Limitations](#limitations) · [Data](#data) · [Quick start](#quick-start) · [Layout](#layout) · [Notes](#notes)

## Situation

A churn model is only as trustworthy as its evaluation. Time-aware features,
a future label window, and a chronological split are what separate a production
model from a homework exercise — and a random split silently leaks a user's
future activity into training.

## Task

Build a leakage-free churn model for a subscription product and judge it by a
business metric — how many real churners a retention team catches in the top 10%
riskiest users — not by AUC alone.

## Actions

- **Time-aware features** computed as-of a snapshot: recency, activity in prior 7/14/30d windows, tenure, trend, sessions, plus categorical channel/device/country/plan — every window strictly *before* the snapshot.
- **Leakage-free label**: churn = no activity in `[snapshot, snapshot+30d]` for users active in the prior 30d.
- **Chronological split**: train (2024-01-15) → val (2024-02-14) → test (2024-03-15), snapshots spaced by `LABEL_WINDOW_DAYS`; `src/config.py` asserts no overlap.
- **Business metric first**: recall@top-decile and lift@top-decile alongside ROC-AUC / PR-AUC.
- **Native TreeSHAP** via LightGBM's `predict_proba(pred_contrib=True)` (no `shap` package); `recency_days` dominates.
- **Leakage claims bound by tests** — deleting post-snapshot activity must leave the feature frame byte-identical.

## Result

- **5.44×** — LightGBM lift@top-decile (recall 0.544); LogReg 5.37× (recall 0.537).
- **0.916 / 0.917** — ROC-AUC LightGBM vs LogReg: a genuine tie, not a LightGBM win.
- **1.75×** — lift once guaranteed negatives are removed (at-risk subgroup, n = 1,692 of 5,581).
- **0.572** — honest churn rate for genuine at-risk customers on the test snapshot.
- **69.7%** — share of the test cohort that is a near-guaranteed negative by construction.

A leakage-free churn model for a subscription product. The point of this
project is the **discipline**, not the algorithm: features are computed "as-of"
a snapshot date, the label is a future 30-day inactivity window, and the split
is chronological (train → val → test snapshots) so the model is evaluated the
way it would be used in production.

## What it does

- **Time-aware features** computed at a snapshot: recency, activity in prior
  7/14/30d windows, tenure, recent trend, average sessions, plus categorical
  channel/device/country/plan. Every window is strictly *before* the snapshot.
- **Leakage-free label**: churn = no activity in `[snapshot, snapshot+30d]` for
  users who were active in the prior 30d (i.e. real at-risk customers).
- **Chronological split**: train (2024-01-15) → val (2024-02-14) → test
  (2024-03-15). No random split — that would leak future activity into training.
  Snapshots are spaced by exactly `LABEL_WINDOW_DAYS`, so label windows cannot
  overlap; `src/config.py` asserts this at import time.
- **Business metric first**: ROC-AUC and PR-AUC, but the decision metric is
  **recall@top-decile** and **lift@top-decile** — if retention acts on the top
  10% riskiest users, how many actual churners do we catch, and how much better
  is that than contacting ten percent of the base at random?
- **Native SHAP** via LightGBM's `predict_proba(pred_contrib=True)` — TreeSHAP
  without the `shap` package. `recency_days` dominates, as expected.
- **Early stopping** on the val snapshot (LightGBM stopped at iteration 58 of 400).

## Results (test snapshot, seed = 42)

| Metric | LightGBM | LogReg |
|---|---|---|
| ROC-AUC | 0.916 | **0.917** |
| PR-AUC | 0.824 | **0.828** |
| Recall@top-decile | **0.544** | 0.537 |
| Lift@top-decile | **5.44×** | 5.37× |
| Brier (calibration) | 0.063 | **0.062** |

`lift@top-decile` is the conventional definition — **precision in the top decile
over the base rate** — not `recall / base_rate`, which is a different quantity
and would report a much smaller number for the same model.

**Honest finding: logistic regression is not beaten here.** It is marginally
ahead on AUC, PR-AUC and Brier and marginally behind on lift. The synthetic
features are nearly linear by construction, so a linear model is the right tool
and the gradient-boosted model earns no trophy. LightGBM is kept as the
*extensible* baseline: it consumes categoricals natively (no one-hot) and exposes
TreeSHAP, which is what makes the feature-attribution section possible at all.
The value of the project is the leakage-free setup and the business metric — the
model comparison is reported as it fell out, not as a headline.

> An earlier version of this README claimed LightGBM was "meaningfully
> better-calibrated". That was an artifact: `class_weight="balanced"` was applied
> to the logistic baseline only, which inflates predicted probabilities and
> pushes its Brier score to 0.099. An honestly-fitted baseline scores 0.062.

Top SHAP features (mean |log-odds contribution|, full top-10 in
`reports/metrics.json`): `recency_days` ≫ `channel` > `recent_trend` > `plan` >
`events_14`.

## At-risk subgroup

The headline metrics above are computed on the full test cohort, and that number
is misleading on its own — see Limitations. Restricted to users who could
actually churn (n = 1,692 of 5,581; base rate 0.572):

| Metric | LightGBM | LogReg |
|---|---|---|
| ROC-AUC | 0.901 | 0.899 |
| PR-AUC | **0.936** | 0.935 |
| Lift@top-decile | 1.75× | 1.74× |
| Brier | 0.163 | **0.157** |

Note how much smaller the lift is once guaranteed negatives are removed: the
5.4× headline is largely earned by trivially separating users who were never
able to churn inside the label window.

## Limitations

- **Most of the cohort is a near-guaranteed negative by construction.** The
  generator caps `death_date` at `HORIZON = 2024-05-01`. Any user whose death is
  at or past a snapshot's label-window end is very unlikely to register 30
  consecutive silent days inside that window. That is 69.7% of the test cohort.
  Churn among them is 0.103% (4 of 3,889), against 57.2% among the rest. Note
  "near-guaranteed", not guaranteed: the mask is a statistical stratifier, and a
  handful of masked users do churn.
- **`base_rate = 0.174` is therefore a mixture artifact**, not a churn rate: an
  average of a ~0.001 rate and a ~0.57 rate. The honest churn rate for genuine
  at-risk customers on the test snapshot is 0.572, and the at-risk table above is
  the one to read. Fixing this properly means changing the generator, which was
  deliberately left alone; instead the censoring mask is exposed as a stratifier
  and documented here.
- **SHAP values are raw-margin (log-odds) contributions**, not probability-space
  ones. They sum to the log-odds prediction and are not additive in probability.
- **Early stopping consumes the val snapshot**, so val is no longer a clean
  held-out set for hyperparameter selection. The test snapshot stays untouched.
- **The at-risk mask uses oracle knowledge** (`death_date`), which would not be
  available at prediction time in production. It is used for *evaluation
  stratification only* — `test_feature_columns_are_allowlisted` binds that it
  never reaches the feature frame.
- Metrics are single-snapshot point estimates with no confidence intervals. The
  honest AUC gap is Δ = −0.0012 (LightGBM minus LogReg), i.e. a genuine tie at
  this sample size; no bootstrap CI was computed for it. An earlier claim that
  the gap was Δ = −0.0127 with 95% CI [−0.019, −0.007] was measured against the
  *handicapped* baseline and does not describe the current code.

## Data

Synthetic, deterministic (seed = 42). 12,000 users sign up over Jun 2023 – Jan
2024; each user has a channel-driven tenure (exponential lifetime) and a
plan-driven daily activity probability. Activity decays slightly approaching
churn. ~455k active user-days.

- `data/users.csv` — attributes + `death_date` (last possible active day).
- `data/activity.csv` — one row per active user-day (`sessions`, `events`).
- `data/generate_data.py` — the generator.

## Quick start

```bash
# Python >=3.10. Deps: pandas, numpy, scikit-learn, lightgbm, matplotlib.
uv run --with pandas --with numpy python data/generate_data.py   # build data
uv run --with pandas --with numpy --with scikit-learn --with lightgbm --with matplotlib python run.py
uv run --with pandas --with numpy --with scikit-learn --with lightgbm --with matplotlib python run.py --no-plot
uv run --with pandas --with numpy --with scikit-learn --with lightgbm --with matplotlib --with pytest pytest -q

# Lint (config lives in pyproject.toml)
uv run --with ruff ruff check .
```

`run.py --no-plot` runs the full pipeline and writes `reports/metrics.json`
without touching matplotlib — useful in a headless environment.

Outputs land in `reports/`: `metrics.json` (all metrics, the at-risk breakdown,
`censored_share`, `best_iteration`, SHAP top-10) + `evaluation.png` (ROC, PR, SHAP bar).

## Layout

```
churn-prediction/
├── data/
│   └── generate_data.py      # synthetic users + activity, seed=42
├── src/
│   ├── __init__.py
│   ├── config.py             # snapshots, windows, label window, model params, split validation
│   ├── features.py           # as-of feature builder, churn label, censoring mask
│   └── model.py              # train/eval/SHAP pipeline, fair baseline
├── tests/
│   ├── conftest.py           # regenerates data when missing or stale
│   └── test_churn.py         # leakage properties, config integrity, metrics, plots
├── reports/                  # metrics.json + evaluation.png (gitignored)
├── pyproject.toml            # deps, ruff config, pytest config
└── run.py                    # CLI (--no-plot)
```

## Notes

- No random train/test split. A random split here would put a user's future
  activity in training and their past in test — silent leakage. The snapshot
  split is the whole reason this project exists.
- The central claim ("features never see the future") is bound by tests, not
  prose: `test_features_ignore_future_activity` (deleting all post-snapshot
  activity must leave the feature frame byte-identical),
  `test_no_activity_on_snapshot_leaks_in`, and
  `test_feature_columns_are_allowlisted` (a frozen column list, so a
  future-window feature cannot be added silently). Both were verified by
  mutation: flipping `< snapshot` to `<= snapshot`, and injecting a
  `future_days` column, each turn the suite red.
- The label deliberately counts a user as churned only if they were active
  recently (prior 30d) and then go silent for 30d. Long-dead users are excluded,
  not labeled — predicting on ghosts is not a real task.
- LightGBM's native TreeSHAP avoids the `shap` package's `llvmlite` dependency,
  which has no wheel for Python 3.14. Same SHAP values, zero extra deps.
- `OneHotEncoder` treats *every* column as categorical unless you scope it. Fed
  the raw frame it expands the ~15 numeric features into 2,935 one-hot columns
  and the baseline collapses to AUC 0.833. Hence the `ColumnTransformer`.
