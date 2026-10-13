"""Figures for the churn-prediction EDA (fan-out, slug=churn).

Light theme, OO style, constrained layout, dpi=300. English labels.

Run:
    EDA="/Users/nikitaboarkin/Desktop/00 ide/00 portfolio/.eda-tools/.venv/bin/python"
    "$EDA" reports/eda/scripts/churn_eda_figures.py
"""

from __future__ import annotations

import pathlib
import warnings

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans", "sans-serif"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
    }
)

HERE = pathlib.Path(__file__).resolve().parent
PROJECT = HERE.parent.parent.parent
DATA = PROJECT / "data"
FIG = HERE.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SNAPSHOT_TEST = pd.Timestamp("2024-03-15")
LABEL_WINDOW_DAYS = 30
PRIOR_ACTIVITY_DAYS = 30
HORIZON = pd.Timestamp("2024-05-01")
SEGMENTS = ["channel", "device", "country", "plan"]


def load():
    users = pd.read_csv(DATA / "users.csv", parse_dates=["signup_date", "death_date"])
    activity = pd.read_csv(DATA / "activity.csv", parse_dates=["active_date"])
    users["lifetime_days"] = (users.death_date - users.signup_date).dt.days
    return users, activity


def snapshot_label(users, activity, snapshot):
    snapshot = pd.Timestamp(snapshot)
    horizon = snapshot + pd.Timedelta(days=LABEL_WINDOW_DAYS)
    prior_start = snapshot - pd.Timedelta(days=PRIOR_ACTIVITY_DAYS)
    cust = activity[
        (activity.active_date >= prior_start) & (activity.active_date < snapshot)
    ].user_id.unique()
    u = users[users.user_id.isin(cust)].copy()
    fut = set(
        activity[
            (activity.active_date >= snapshot) & (activity.active_date < horizon)
        ].user_id.unique()
    )
    u["churn"] = (~u.user_id.isin(fut)).astype(int)
    u["censored"] = (pd.to_datetime(u.death_date) >= horizon).to_numpy()
    return u.reset_index(drop=True)


# ---------------------------------------------------------------- fig 1
def fig_lifetime(users, out):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    d = users.lifetime_days
    axs[0].hist(d, bins=50, color="#4C72B0", edgecolor="white")
    axs[0].axvline(d.median(), color="#C44E52", lw=2, label=f"median = {d.median():.0f}d")
    axs[0].axvline(d.mean(), color="#DD8452", lw=2, ls="--", label=f"mean = {d.mean():.0f}d")
    axs[0].set(
        xlabel="Lifetime (days)", ylabel="Users", title="User lifetime distribution (linear)"
    )
    axs[0].legend(frameon=False)

    bins = np.logspace(np.log10(max(d.min(), 1)), np.log10(d.max()), 40)
    axs[1].hist(d.clip(lower=1), bins=bins, color="#55A868", edgecolor="white")
    axs[1].set_xscale("log")
    axs[1].set(
        xlabel="Lifetime (days, log scale)",
        ylabel="Users",
        title=f"Lifetime on log axis (skew = {d.skew():.2f})",
    )
    fig.suptitle("Derived lifetime_days — right-censored at horizon 2024-05-01", fontsize=13)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig_churn_by_segment(t, out):
    overall = t.churn.mean()
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for ax, col in zip(axs.ravel(), SEGMENTS):
        g = t.groupby(col)["churn"].mean().sort_values(ascending=False)
        bars = ax.bar(g.index, g.values * 100, color="#4C72B0", edgecolor="black")
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
        ax.axhline(
            overall * 100, color="#C44E52", ls="--", lw=1.5, label=f"overall {overall * 100:.1f}%"
        )
        ax.set(xlabel=col, ylabel="Churn rate, %", title=f"Churn rate by {col}")
        ax.set_ylim(0, g.values.max() * 100 * 1.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Churn rate by segment (test snapshot 2024-03-15, n=5581) — channel has the widest gap",
        fontsize=13,
    )
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig_trend(activity, out):
    act = activity.copy()
    act["week"] = act.active_date.dt.to_period("W").dt.start_time
    wk = act.groupby("week").agg(active_days=("active_date", "count"), sessions=("sessions", "sum"))
    wk["sessions_per_day"] = wk.sessions / wk.active_days

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), layout="constrained", sharex=True)
    axs[0].plot(wk.index, wk.active_days, marker="o", ms=3, color="#4C72B0")
    axs[0].set(
        ylabel="Active user-days / week",
        title="Weekly activity — flow (volume) vs behaviour (per active day)",
    )
    last = wk.index[-1]
    axs[0].axvspan(last, last + pd.Timedelta(days=7), color="#C44E52", alpha=0.12)
    axs[0].annotate(
        "last week partial\n(data ends 2024-05-01)",
        xy=(last, wk.active_days.iloc[-1]),
        xytext=(-140, 30),
        textcoords="offset points",
        fontsize=8,
        color="#C44E52",
        arrowprops=dict(arrowstyle="->", color="#C44E52"),
    )

    axs[1].plot(wk.index, wk.sessions_per_day, marker="o", ms=3, color="#55A868")
    axs[1].set(
        xlabel="Week",
        ylabel="Sessions / active user-day",
        title="Behaviour is flat (~2.50) — all weekly movement is flow",
    )
    axs[1].set_ylim(2.3, 2.7)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4
def fig_censoring(t, out):
    cens = t[t.censored]
    risk = t[~t.censored]
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    labels = [
        "Censored\n(no churn observable)",
        "At-risk\n(can churn)",
        "Full test cohort\n(mixture)",
    ]
    vals = [cens.churn.mean() * 100, risk.churn.mean() * 100, t.churn.mean() * 100]
    ns = [len(cens), len(risk), len(t)]
    bars = ax.bar(labels, vals, color=["#8C8C8C", "#C44E52", "#4C72B0"], edgecolor="black")
    for b, v, n in zip(bars, vals, ns):
        ax.annotate(
            f"{v:.1f}%\n(n={n:,})",
            (b.get_x() + b.get_width() / 2, v),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set(
        ylabel="Churn rate, %",
        title="The honest churn rate: the 17.4% base rate is a mixture",
        ylim=(0, 75),
    )
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 5
def fig_paradox(t, out):
    g = t.groupby("channel").agg(
        churn_full=("churn", "mean"), at_risk_share=("censored", lambda s: 1 - s.mean())
    )
    g["churn_at_risk"] = t[~t.censored].groupby("channel")["churn"].mean()
    g = g.sort_values("churn_full", ascending=False)

    x = np.arange(len(g))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    b1 = ax.bar(
        x - w / 2, g.churn_full * 100, w, label="Full cohort", color="#4C72B0", edgecolor="black"
    )
    b2 = ax.bar(
        x + w / 2,
        g.churn_at_risk * 100,
        w,
        label="At-risk only",
        color="#C44E52",
        edgecolor="black",
    )
    ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=8, padding=2)
    ax.set(
        xticks=x,
        xticklabels=g.index,
        xlabel="channel",
        ylabel="Churn rate, %",
        title="Simpson's paradox: the channel 'effect' is a tenure/censoring artefact",
        ylim=(0, 80),
    )
    ax.legend(frameon=False, loc="upper right")

    ax2 = ax.twinx()
    ax2.plot(x, g.at_risk_share * 100, color="#55A868", marker="D", lw=2, label="at-risk share")
    ax2.set(ylabel="Share of channel that is at-risk, %", ylim=(0, 70))
    ax2.legend(frameon=False, loc="center right")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    users, activity = load()
    t = snapshot_label(users, activity, SNAPSHOT_TEST)
    fig_lifetime(users, FIG / "fig1_lifetime_distribution.png")
    fig_churn_by_segment(t, FIG / "fig2_churn_by_segment.png")
    fig_trend(activity, FIG / "fig3_activity_trend.png")
    fig_censoring(t, FIG / "fig4_censoring_stratification.png")
    fig_paradox(t, FIG / "fig5_channel_paradox.png")
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
