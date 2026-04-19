"""Tests for the cost-aware threshold search."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_trainer import find_cost_optimal_threshold


def test_threshold_decreases_when_false_approves_penalised_more() -> None:
    """If missing a defaulter is expensive, the optimal threshold moves DOWN
    (we should deny more)."""
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 1000))
    probs = np.clip(rng.normal(y.values.astype(float) * 0.3 + 0.4, 0.15), 0, 1)

    t_balanced = find_cost_optimal_threshold(y, probs, cost_fa=1.0, cost_fd=1.0)
    t_fa_heavy = find_cost_optimal_threshold(y, probs, cost_fa=10.0, cost_fd=1.0)
    assert t_fa_heavy <= t_balanced + 1e-6, (
        f"expected FA-heavy threshold <= balanced threshold, got {t_fa_heavy} vs {t_balanced}"
    )


def test_threshold_increases_when_false_denies_penalised_more() -> None:
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 1000))
    probs = np.clip(rng.normal(y.values.astype(float) * 0.3 + 0.4, 0.15), 0, 1)

    t_balanced = find_cost_optimal_threshold(y, probs, cost_fa=1.0, cost_fd=1.0)
    t_fd_heavy = find_cost_optimal_threshold(y, probs, cost_fa=1.0, cost_fd=10.0)
    assert t_fd_heavy >= t_balanced - 1e-6, (
        f"expected FD-heavy threshold >= balanced threshold, got {t_fd_heavy} vs {t_balanced}"
    )
