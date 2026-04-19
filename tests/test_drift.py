"""Tests for the PSI drift module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.drift import psi_for_feature, psi_report


def test_psi_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    # Self-comparison should yield ~0; allow a little noise from quantile ties
    psi = psi_for_feature(x, x)
    assert psi == 0.0 or abs(psi) < 1e-4


def test_psi_large_for_shifted_distributions() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.0, size=5000)
    y = rng.normal(loc=3.0, size=5000)  # strong shift
    psi = psi_for_feature(x, y)
    assert psi > 0.25, f"expected PSI > 0.25 for strongly shifted distributions, got {psi}"


def test_psi_handles_nans() -> None:
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    psi = psi_for_feature(x, y)
    assert np.isfinite(psi)


def test_psi_report_returns_all_features_with_severity_col() -> None:
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(size=1000)})
    cur = pd.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(loc=2.0, size=1000)})
    report = psi_report(ref, cur)
    assert set(report.columns) == {"feature", "psi", "severity"}
    assert set(report["feature"]) == {"a", "b"}
    # feature 'b' has a large shift
    assert report.loc[report["feature"] == "b", "severity"].iloc[0] in ("MODERATE", "MAJOR")
