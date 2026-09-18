"""CLI entry point: train/evaluate the churn pipeline.

Usage:
    uv run python run.py            # full pipeline (train on train snapshot, eval on test)
    uv run python run.py --no-plot   # skip matplotlib
"""
from __future__ import annotations

import argparse

from src.model import run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Churn prediction pipeline.")
    ap.add_argument("--no-plot", action="store_true", help="skip saving plots")
    args = ap.parse_args()
    run_pipeline(plot=not args.no_plot)


if __name__ == "__main__":
    main()
