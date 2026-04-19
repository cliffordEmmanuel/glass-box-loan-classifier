"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_ingestion import TARGET_COL


@pytest.fixture
def tiny_raw_df() -> pd.DataFrame:
    """A tiny fake credit dataset with all the quirks the real one has.

    Includes: a missing MonthlyIncome, a sentinel (96) in a past-due column,
    and an outlier in DebtRatio and RevolvingUtilization.
    """
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        TARGET_COL: rng.integers(0, 2, n),
        "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 0.8, n),
        "age": rng.integers(18, 85, n),
        "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 3, n),
        "DebtRatio": rng.uniform(0, 2, n),
        "MonthlyIncome": rng.uniform(1000, 10000, n),
        "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 15, n),
        "NumberOfTimes90DaysLate": rng.integers(0, 2, n),
        "NumberRealEstateLoansOrLines": rng.integers(0, 5, n),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 2, n),
        "NumberOfDependents": rng.integers(0, 5, n),
    })
    # Inject the data quirks
    df.loc[0, "MonthlyIncome"] = np.nan
    df.loc[1, "NumberOfDependents"] = np.nan
    df.loc[2, "NumberOfTime30-59DaysPastDueNotWorse"] = 96  # sentinel
    df.loc[3, "NumberOfTimes90DaysLate"] = 98  # sentinel
    df.loc[4, "DebtRatio"] = 1e5  # outlier
    df.loc[5, "RevolvingUtilizationOfUnsecuredLines"] = 50  # outlier
    return df
