"""Continuous drift detector module for King County House Price System."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root and dags to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "dags")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml import monitoring


def detect_drift() -> dict:
    """Run drift detection against serving model and return report."""
    s_info = monitoring.serving_info()
    report = monitoring.check(s_info)
    print(f"[DriftDetector] Drift report: needs_retrain={report.get('needs_retrain')}, MAPE={report.get('mape')}")
    return report


if __name__ == "__main__":
    rep = detect_drift()
    print("Report:", rep)
