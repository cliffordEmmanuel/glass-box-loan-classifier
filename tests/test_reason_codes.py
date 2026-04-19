"""Tests for adverse-action reason codes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.reason_codes import generate_reason_codes


def _mock_shap(feat_names: list[str], values: list[float]) -> SimpleNamespace:
    """Mock a per-instance shap.Explanation."""
    return SimpleNamespace(
        feature_names=feat_names,
        values=np.asarray(values, dtype=float),
    )


def test_only_adverse_features_surfaced() -> None:
    # Two features push toward denial (positive SHAP), one toward approval.
    feats = [
        "RevolvingUtilizationOfUnsecuredLines",
        "MonthlyIncome",
        "age",
    ]
    sv = _mock_shap(feats, [0.3, 0.5, -0.2])
    instance = pd.Series({
        "RevolvingUtilizationOfUnsecuredLines": 0.85,
        "MonthlyIncome": 2200,
        "age": 45,
    })

    reasons = generate_reason_codes(sv, instance, top_k=3)
    # Only the two positive features should appear
    names = {r.feature for r in reasons}
    assert names == {"RevolvingUtilizationOfUnsecuredLines", "MonthlyIncome"}


def test_reasons_ordered_by_impact() -> None:
    feats = ["DebtRatio", "RevolvingUtilizationOfUnsecuredLines", "NumberOfTimes90DaysLate"]
    sv = _mock_shap(feats, [0.1, 0.4, 0.25])
    instance = pd.Series({
        "DebtRatio": 0.6,
        "RevolvingUtilizationOfUnsecuredLines": 0.9,
        "NumberOfTimes90DaysLate": 2,
    })
    reasons = generate_reason_codes(sv, instance, top_k=3)
    # Most damaging first
    assert reasons[0].feature == "RevolvingUtilizationOfUnsecuredLines"
    assert reasons[1].feature == "NumberOfTimes90DaysLate"
    assert reasons[2].feature == "DebtRatio"


def test_top_k_caps_output() -> None:
    feats = [f"f{i}" for i in range(6)]
    sv = _mock_shap(feats, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    instance = pd.Series({f: 1.0 for f in feats})
    reasons = generate_reason_codes(sv, instance, top_k=3)
    assert len(reasons) == 3


def test_empty_when_all_contributions_negative() -> None:
    feats = ["RevolvingUtilizationOfUnsecuredLines", "MonthlyIncome"]
    sv = _mock_shap(feats, [-0.1, -0.2])
    instance = pd.Series({"RevolvingUtilizationOfUnsecuredLines": 0.1, "MonthlyIncome": 9000})
    reasons = generate_reason_codes(sv, instance, top_k=3)
    assert reasons == []


def test_favourable_direction_surfaces_negative_shap_only() -> None:
    """``direction='favourable'`` should mirror the adverse logic: only
    features with *negative* SHAP (i.e. pushing toward approval) appear,
    ranked most-helpful first."""
    feats = ["MonthlyIncome", "RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]
    sv = _mock_shap(feats, [-0.4, -0.1, 0.2])  # DebtRatio is adverse
    instance = pd.Series({
        "MonthlyIncome": 9500,
        "RevolvingUtilizationOfUnsecuredLines": 0.12,
        "DebtRatio": 0.6,
    })
    reasons = generate_reason_codes(sv, instance, top_k=3, direction="favourable")
    names = [r.feature for r in reasons]
    # DebtRatio (positive SHAP) must not appear; most-helpful ranks first.
    assert names == ["MonthlyIncome", "RevolvingUtilizationOfUnsecuredLines"]
    # Phrasing should be the favourable template, not the adverse one.
    assert "Strong monthly income" in reasons[0].description


def test_invalid_direction_raises() -> None:
    sv = _mock_shap(["MonthlyIncome"], [-0.1])
    instance = pd.Series({"MonthlyIncome": 8000})
    try:
        generate_reason_codes(sv, instance, direction="sideways")
    except ValueError:
        return
    raise AssertionError("expected ValueError for bad direction")
