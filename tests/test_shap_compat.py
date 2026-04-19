"""Tests for the SHAP/XGBoost compatibility shim."""

from __future__ import annotations

import builtins

from src.shap_compat import _safe_float, shap_xgb_compat


def test_safe_float_parses_plain_numbers() -> None:
    assert _safe_float("0.5") == 0.5
    assert _safe_float(0.5) == 0.5
    assert _safe_float(3) == 3.0


def test_safe_float_parses_bracketed_scientific() -> None:
    assert _safe_float("[5E-1]") == 0.5
    assert _safe_float("[4.5E-1]") == 0.45


def test_context_manager_restores_float() -> None:
    original = builtins.float
    with shap_xgb_compat():
        assert builtins.float is not original
    assert builtins.float is original


def test_context_manager_restores_on_exception() -> None:
    original = builtins.float
    try:
        with shap_xgb_compat():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert builtins.float is original
