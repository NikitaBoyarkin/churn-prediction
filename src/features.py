"""Leakage-free feature engineering "as-of" a snapshot date.

All features use activity strictly BEFORE the snapshot. The churn label uses
activity in the future window [snapshot, snapshot + LABEL_WINDOW_DAYS]. A user
is included only if they had >=1 active day in the prior PRIOR_ACTIVITY_DAYS
(i.e. they are an active customer worth predicting on).
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def load_raw():
    here = pathlib.Path(__file__).parent.parent / "data"
    users = pd.read_csv(here / "users.csv", parse_dates=["signup_date", "death_date"])
    activity = pd.read_csv(here / "activity.csv", parse_dates=["active_date"])
    return users, activity


def _label_and_filter(users, activity, snapshot):
    snapshot = pd.Timestamp(snapshot)
    horizon = snapshot + pd.Timedelta(days=C.LABEL_WINDOW_DAYS)

    # Customer: had >=1 active day in the prior window.
    prior_start = snapshot - pd.Timedelta(days=C.PRIOR_ACTIVITY_DAYS)
    prior = activity[(activity.active_date >= prior_start) & (activity.active_date < snapshot)]
    customers = prior["user_id"].unique()
    u = users[users.user_id.isin(customers)].copy()
    if u.empty:
        return u.assign(churn=pd.Series(dtype=int))

    # Churn = no active day in [snapshot, snapshot+window].
    future = activity[(activity.active_date >= snapshot) & (activity.active_date < horizon)]
    active_future = set(future["user_id"].unique())
    u["churn"] = (~u["user_id"].isin(active_future)).astype(int)
    # No death_date exclusion: a user whose last active day is in [snapshot-30, snapshot)
    # is exactly a churner (active recently, then silent for 30d). The prior-window
    # filter already removes users who died long before snapshot.
    return u.reset_index(drop=True)


def _features(users, activity, snapshot):
    snapshot = pd.Timestamp(snapshot)
    prior30 = snapshot - pd.Timedelta(days=30)
    prior14 = snapshot - pd.Timedelta(days=14)
    prior7 = snapshot - pd.Timedelta(days=7)

    a = activity[activity.active_date < snapshot].copy()
    g = a.groupby("user_id")

    last_active = a.groupby("user_id")["active_date"].max()
    feat = pd.DataFrame({"user_id": users["user_id"].values})
    feat = feat.merge(last_active.rename("last_active"), on="user_id", how="left")

    def days_in(win_start, win_end):
        sub = a[(a.active_date >= win_start) & (a.active_date < win_end)]
        agg = sub.groupby("user_id").agg(
            active_days=("active_date", "nunique"),
            sessions=("sessions", "sum"),
            events=("events", "sum"),
        ).reset_index()
        return agg

    f30 = days_in(prior30, snapshot).rename(columns=lambda c: c if c == "user_id" else c + "_30")
    f14 = days_in(prior14, snapshot).rename(columns=lambda c: c if c == "user_id" else c + "_14")
    f7 = days_in(prior7, snapshot).rename(columns=lambda c: c if c == "user_id" else c + "_7")

    for dfx in (f30, f14, f7):
        feat = feat.merge(dfx, on="user_id", how="left")

    feat = feat.merge(users[["user_id", "signup_date"]], on="user_id", how="left")
    feat["tenure_days"] = (snapshot - feat["signup_date"]).dt.days
    feat = feat.drop(columns=["signup_date"])
    feat["recency_days"] = (snapshot - feat["last_active"]).dt.days
    feat.fillna({"active_days_30": 0, "active_days_14": 0, "active_days_7": 0,
                 "sessions_30": 0, "events_30": 0,
                 "sessions_14": 0, "events_14": 0,
                 "sessions_7": 0, "events_7": 0}, inplace=True)
    feat["recency_days"] = feat["recency_days"].fillna(feat["tenure_days"])
    feat["active_ratio_30"] = feat["active_days_30"] / 30.0
    feat["recency_ratio"] = feat["recency_days"] / (feat["tenure_days"] + 1)
    feat["recent_trend"] = (feat["active_days_7"] / 7.0) / (feat["active_days_30"] / 30.0 + 1e-9)
    feat["avg_sessions_active_day"] = feat["sessions_30"] / (feat["active_days_30"] + 1e-9)

    # categorical attrs
    feat = feat.merge(users[["user_id"] + C.CATEGORICAL], on="user_id", how="left")
    feat = feat.drop(columns=["last_active"])
    return feat


def build_dataset(snapshot):
    users, activity = load_raw()
    u = _label_and_filter(users, activity, snapshot)
    if u.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    y = u["churn"]
    X = _features(u, activity, snapshot)
    return X, y


import pathlib  # used in load_raw
