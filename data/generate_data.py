"""Deterministic synthetic data generator for churn prediction.

Models a subscription product: users sign up, are active on some days, and
eventually churn (stop being active). The generator emits a long-format
activity table (one row per active user-day) plus a users table, so features can
be computed "as-of" any snapshot date without leakage.

Reproducible: seed=42. Run:
    uv run python data/generate_data.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
SEED = 42
N_USERS = 12_000
SIGNUP_START = pd.Timestamp("2023-06-01")
SIGNUP_END = pd.Timestamp("2024-01-31")
HORIZON = pd.Timestamp("2024-05-01")  # data ends here; needed for test label window

CHANNELS = ["organic", "paid_search", "social", "referral", "email"]
CHANNEL_PROB = [0.30, 0.25, 0.20, 0.15, 0.10]
# mean tenure in days (exponential scale) per channel — drives monthly churn rate
CHANNEL_TENURE = {"organic": 200, "paid_search": 80, "social": 120,
                  "referral": 350, "email": 150}
DEVICES = ["ios", "android", "web"]
DEVICE_PROB = [0.4, 0.4, 0.2]
COUNTRIES = ["RU", "UA", "KZ", "BY", "Other"]
COUNTRY_PROB = [0.55, 0.12, 0.12, 0.08, 0.13]
PLANS = ["free", "basic", "pro"]
PLAN_PROB = [0.6, 0.3, 0.1]
# daily activity probability while alive, by plan
PLAN_ACTIVE = {"free": 0.30, "basic": 0.45, "pro": 0.65}


def build_users(rng: np.random.Generator) -> pd.DataFrame:
    uids = np.arange(1, N_USERS + 1)
    span = (SIGNUP_END - SIGNUP_START).days
    signup = SIGNUP_START + pd.to_timedelta(rng.integers(0, span + 1, N_USERS), unit="D")
    channel = rng.choice(CHANNELS, N_USERS, p=CHANNEL_PROB)
    device = rng.choice(DEVICES, N_USERS, p=DEVICE_PROB)
    country = rng.choice(COUNTRIES, N_USERS, p=COUNTRY_PROB)
    plan = rng.choice(PLANS, N_USERS, p=PLAN_PROB)
    # Tenure: exponential-ish lifetime in days, scaled by channel quality.
    scale = np.array([CHANNEL_TENURE[c] for c in channel])
    tenure = (rng.exponential(scale=scale)).astype(int) + 5
    death = pd.to_datetime(signup) + pd.to_timedelta(tenure, unit="D")
    death = np.minimum(death, HORIZON)  # cap at horizon (right-censored)
    return pd.DataFrame({
        "user_id": uids,
        "signup_date": signup.normalize().date,
        "channel": channel,
        "device": device,
        "country": country,
        "plan": plan,
        "death_date": pd.to_datetime(death).normalize().date,
    })


def build_activity(rng: np.random.Generator, users: pd.DataFrame) -> pd.DataFrame:
    """One row per active user-day, with sessions and events that day."""
    rows_u, rows_d, rows_s, rows_e = [], [], [], []
    for _, u in users.iterrows():
        p_active = PLAN_ACTIVE[u["plan"]]
        start = pd.Timestamp(u["signup_date"])
        end = min(pd.Timestamp(u["death_date"]), HORIZON)
        if end <= start:
            continue
        days = pd.date_range(start, end, freq="D")
        active = rng.random(len(days)) < p_active
        # Activity slightly decays approaching death for realism.
        decay = 1.0 - np.linspace(0, 0.25, len(days))
        active = active & (rng.random(len(days)) < decay)
        if not active.any():
            continue
        for d, on in zip(days, active):
            if not on:
                continue
            s = rng.integers(1, 5)  # sessions that day
            e = s * rng.integers(1, 6)  # events
            rows_u.append(u["user_id"])
            rows_d.append(d)
            rows_s.append(int(s))
            rows_e.append(int(e))
    activity = pd.DataFrame({
        "user_id": rows_u, "active_date": rows_d,
        "sessions": rows_s, "events": rows_e,
    }).sort_values(["user_id", "active_date"]).reset_index(drop=True)
    return activity


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"Generating churn dataset (seed={SEED}, users={N_USERS})...")
    users = build_users(rng)
    activity = build_activity(rng, users)
    users.to_csv(HERE / "users.csv", index=False)
    activity.to_csv(HERE / "activity.csv", index=False)
    print(f"  users: {len(users):,} rows")
    print(f"  activity: {len(activity):,} active user-days")
    print(f"  date range: {activity.active_date.min().date()} .. {activity.active_date.max().date()}")
    print(f"Done. users.csv + activity.csv in {HERE}")


if __name__ == "__main__":
    main()
