"""Ensure the synthetic dataset exists before running tests."""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent.parent
ACTIVITY = ROOT / "data" / "activity.csv"


def pytest_configure():
    if not ACTIVITY.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "data" / "generate_data.py")], check=True
        )
