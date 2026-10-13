"""EDA analysis for churn-prediction (fan-out, slug=churn).

Recomputes every data-level README claim from data/users.csv + data/activity.csv
and dumps a JSON summary. Model metrics (lift/AUC) are NOT re-run here.

Run:
    EDA="/Users/nikitaboarkin/Desktop/00 ide/00 portfolio/.eda-tools/.venv/bin/python"
    "$EDA" reports/eda/scripts/churn_eda_analysis.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
PROJECT = HERE.parent.parent.parent  # reports/eda/scripts -> project root
DATA = PROJECT / "data"
OUT_JSON = HERE / "churn_eda_numbers.json"

SNAPSHOT_TEST = pd.Timestamp("2024-03-15")
LABEL_WINDOW_DAYS = 30
PRIOR_ACTIVITY_DAYS = 30
HORIZON = pd.Timestamp("2024-05-01")

SEGMENTS = ["channel", "device", "country", "plan"]


def load():
    users = pd.read_csv(DATA / "users.csv", parse_dates=["signup_date", "death_date"])
    activity = pd.read_csv(DATA / "activity.csv", parse_dates=["active_date"])
    return users, activity


def snapshot_label(users, activity, snapshot):
    """Replicates src/features.py: customers active in prior 30d, churn = silent 30d after."""
    snapshot = pd.Timestamp(snapshot)
    horizon = snapshot + pd.Timedelta(days=LABEL_WINDOW_DAYS)
    prior_start = snapshot - pd.Timedelta(days=PRIOR_ACTIVITY_DAYS)
    prior = activity[(activity.active_date >= prior_start) & (activity.active_date < snapshot)]
    customers = prior["user_id"].unique()
    u = users[users.user_id.isin(customers)].copy()
    future = activity[(activity.active_date >= snapshot) & (activity.active_date < horizon)]
    active_future = set(future["user_id"].unique())
    u["churn"] = (~u["user_id"].isin(active_future)).astype(int)
    label_end = snapshot + pd.Timedelta(days=LABEL_WINDOW_DAYS)
    u["censored"] = (pd.to_datetime(u["death_date"]) >= label_end).to_numpy()
    return u.reset_index(drop=True)


def main():
    users, activity = load()
    out: dict = {}

    # ---- Dataset shape (README: 12,000 users, ~455k active user-days) ----
    out["users_rows"] = int(len(users))
    out["users_cols"] = int(users.shape[1])
    out["activity_rows"] = int(len(activity))
    out["signup_min"] = str(users.signup_date.min().date())
    out["signup_max"] = str(users.signup_date.max().date())
    out["death_min"] = str(users.death_date.min().date())
    out["death_max"] = str(users.death_date.max().date())
    out["activity_date_min"] = str(activity.active_date.min().date())
    out["activity_date_max"] = str(activity.active_date.max().date())

    # ---- Quality / validity ----
    out["users_full_dup"] = int(users.duplicated().sum())
    out["activity_full_dup"] = int(activity.duplicated().sum())
    out["users_dup_id"] = int(users.user_id.duplicated().sum())
    out["activity_dup_userday"] = int(activity.duplicated(["user_id", "active_date"]).sum())
    out["users_null_counts"] = {c: int(users[c].isna().sum()) for c in users.columns}
    out["activity_null_counts"] = {c: int(activity[c].isna().sum()) for c in activity.columns}
    # FK integrity: every activity user_id exists in users
    orphan_ids = set(activity.user_id.unique()) - set(users.user_id.unique())
    out["activity_orphan_users"] = int(len(orphan_ids))
    out["death_at_horizon"] = int((users.death_date == HORIZON).sum())
    out["death_before_signup"] = int((users.death_date < users.signup_date).sum())
    out["sessions_range"] = [int(activity.sessions.min()), int(activity.sessions.max())]
    out["events_range"] = [int(activity.events.min()), int(activity.events.max())]

    # ---- Derived tenure (lifetime_days), right-censored at horizon ----
    users = users.copy()
    users["lifetime_days"] = (users.death_date - users.signup_date).dt.days
    out["lifetime_days_mean"] = float(users.lifetime_days.mean())
    out["lifetime_days_median"] = float(users.lifetime_days.median())
    out["lifetime_days_skew"] = float(users.lifetime_days.skew())
    out["lifetime_days_p95"] = float(users.lifetime_days.quantile(0.95))
    # naive is_churned = death_date not null (spec simplification) -> all rows
    out["naive_is_churned_share"] = float(users.death_date.notna().mean())

    # ---- Snapshot label (the real project definition) ----
    t = snapshot_label(users, activity, SNAPSHOT_TEST)
    out["test_n"] = int(len(t))
    out["test_base_rate"] = float(t.churn.mean())
    out["test_censored_n"] = int(t.censored.sum())
    out["test_censored_share"] = float(t.censored.mean())
    at_risk = t[~t.censored]
    censored = t[t.censored]
    out["test_at_risk_n"] = int(len(at_risk))
    out["test_at_risk_base_rate"] = float(at_risk.churn.mean())
    out["test_censored_churn_rate"] = float(censored.churn.mean())
    out["test_censored_churners"] = int(censored.churn.sum())
    out["test_at_risk_churners"] = int(at_risk.churn.sum())

    # ---- Metric by segment: churn rate on test cohort (at-risk + full) ----
    seg = {}
    for c in SEGMENTS:
        g = t.groupby(c)["churn"].agg(["count", "mean"])
        g["share"] = g["count"] / g["count"].sum()
        g["dev_pp"] = (g["mean"] - t.churn.mean()) * 100
        seg[c] = {
            k: {
                "count": int(v["count"]),
                "churn_rate": float(v["mean"]),
                "share": float(v["share"]),
                "dev_pp": float(v["dev_pp"]),
            }
            for k, v in g.iterrows()
        }
    out["churn_by_segment_full"] = seg
    out["overall_churn_rate"] = float(t.churn.mean())

    # max-gap segment (ratio max/min churn rate), full cohort
    gap = {}
    for c in SEGMENTS:
        rates = pd.Series({k: v["churn_rate"] for k, v in seg[c].items()})
        gap[c] = {
            "max_seg": str(rates.idxmax()),
            "min_seg": str(rates.idxmin()),
            "ratio": float(rates.max() / rates.min()) if rates.min() > 0 else None,
        }
    out["segment_gap_full"] = gap

    # at-risk only segment churn (cleaner)
    seg_ar = {}
    for c in SEGMENTS:
        g = at_risk.groupby(c)["churn"].agg(["count", "mean"])
        seg_ar[c] = {
            k: {"count": int(v["count"]), "churn_rate": float(v["mean"])} for k, v in g.iterrows()
        }
    out["churn_by_segment_at_risk"] = seg_ar

    # ---- Simpson check: is channel gap a composition artefact of plan/tenure? ----
    ct = pd.crosstab(t.channel, t.plan, normalize="index")
    out["channel_plan_composition"] = {
        ch: {p: float(v) for p, v in row.items()} for ch, row in ct.iterrows()
    }

    # ---- Activity trend: flow vs behaviour ----
    act = activity.copy()
    act["week"] = act.active_date.dt.to_period("W").dt.start_time
    wk = act.groupby("week").agg(
        active_days=("active_date", "count"), sessions=("sessions", "sum"), events=("events", "sum")
    )
    wk["users"] = act.groupby("week")["user_id"].nunique()
    wk["sessions_per_active_day"] = wk["sessions"] / wk["active_days"]
    wk["users_per_active_day"] = wk["users"] / wk["active_days"]
    out["weekly_last8"] = [
        {
            "week": str(i.date()),
            "active_days": int(r.active_days),
            "users": int(r.users),
            "sessions_per_active_day": float(r.sessions_per_active_day),
        }
        for i, r in wk.tail(8).iterrows()
    ]
    out["weekly_n"] = int(len(wk))
    out["active_users_total"] = int(activity.user_id.nunique())
    out["active_days_per_user_mean"] = float(len(activity) / activity.user_id.nunique())

    # correlation sessions~events
    out["corr_sessions_events_pearson"] = float(activity.sessions.corr(activity.events))
    out["corr_sessions_events_spearman"] = float(
        activity.sessions.corr(activity.events, method="spearman")
    )

    # ---- day-of-week pattern ----
    dow = activity.groupby(activity.active_date.dt.dayofweek)["sessions"].agg(["count", "mean"])
    out["dow_active_days"] = {int(k): int(v["count"]) for k, v in dow.iterrows()}

    # ---- snapshot cohort sizes for all splits (README n: 6768/6597/5581) ----
    out["split_n"] = {
        name: int(len(snapshot_label(users, activity, s)))
        for name, s in {"train": "2024-01-15", "val": "2024-02-14", "test": "2024-03-15"}.items()
    }

    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
