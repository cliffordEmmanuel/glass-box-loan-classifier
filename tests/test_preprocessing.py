"""Preprocessing tests — including a regression test for the leakage bug."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_ingestion import TARGET_COL
from src.preprocessing import (
    OUTLIER_CAPS,
    PAST_DUE_SENTINELS,
    FeatureScaler,
    Preprocessor,
    get_feature_matrix,
)


def test_preprocessor_imputes_missing(tiny_raw_df: pd.DataFrame) -> None:
    pre = Preprocessor().fit(tiny_raw_df)
    out = pre.transform(tiny_raw_df)
    assert out["MonthlyIncome"].isna().sum() == 0
    assert out["NumberOfDependents"].isna().sum() == 0


def test_preprocessor_caps_outliers(tiny_raw_df: pd.DataFrame) -> None:
    pre = Preprocessor().fit(tiny_raw_df)
    out = pre.transform(tiny_raw_df)
    assert out["DebtRatio"].max() <= OUTLIER_CAPS["DebtRatio"]
    assert out["RevolvingUtilizationOfUnsecuredLines"].max() <= OUTLIER_CAPS[
        "RevolvingUtilizationOfUnsecuredLines"
    ]


def test_preprocessor_replaces_past_due_sentinels(tiny_raw_df: pd.DataFrame) -> None:
    pre = Preprocessor().fit(tiny_raw_df)
    out = pre.transform(tiny_raw_df)
    for col in [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ]:
        assert not out[col].isin(PAST_DUE_SENTINELS).any()


def test_preprocessor_no_leakage_between_fit_and_transform(tiny_raw_df: pd.DataFrame) -> None:
    """Regression test: the preprocessor must use train statistics when
    transforming a different split, not the other split's own medians.

    Construct a train split whose MonthlyIncome median is very different
    from the test split's median. After fit(train) -> transform(test), the
    NaN in test should be filled with the *train* median, not the test one.
    """
    train = tiny_raw_df.copy()
    train.loc[:, "MonthlyIncome"] = 1000.0  # train median will be 1000
    test = tiny_raw_df.copy()
    test.loc[:, "MonthlyIncome"] = 9000.0   # test median would be 9000
    test.loc[0, "MonthlyIncome"] = np.nan   # one missing row

    pre = Preprocessor().fit(train)
    out = pre.transform(test)
    # Should be filled with the TRAIN median (1000), not the test median (9000).
    assert out.loc[0, "MonthlyIncome"] == 1000.0


def test_preprocessor_transform_before_fit_raises() -> None:
    pre = Preprocessor()
    try:
        pre.transform(pd.DataFrame({"MonthlyIncome": [1.0]}))
    except RuntimeError:
        return
    raise AssertionError("Preprocessor.transform before fit should raise RuntimeError")


def test_preprocessor_is_idempotent_on_transform(tiny_raw_df: pd.DataFrame) -> None:
    """Transforming twice should give the same result."""
    pre = Preprocessor().fit(tiny_raw_df)
    first = pre.transform(tiny_raw_df)
    second = pre.transform(first)
    pd.testing.assert_frame_equal(first, second)


def test_feature_scaler_fit_transform_mean_near_zero(tiny_raw_df: pd.DataFrame) -> None:
    pre = Preprocessor().fit(tiny_raw_df)
    clean = pre.transform(tiny_raw_df)
    X, _ = get_feature_matrix(clean)
    fs = FeatureScaler().fit(X)
    Xs = fs.transform(X)
    assert np.allclose(Xs.mean().values, 0, atol=1e-6)
    assert np.allclose(Xs.std(ddof=0).values, 1, atol=1e-6)


def test_feature_scaler_preserves_column_order(tiny_raw_df: pd.DataFrame) -> None:
    pre = Preprocessor().fit(tiny_raw_df)
    clean = pre.transform(tiny_raw_df)
    X, _ = get_feature_matrix(clean)
    fs = FeatureScaler().fit(X)
    shuffled = X[X.columns[::-1]]
    # transform should re-order columns back to the fitted order
    Xs = fs.transform(shuffled)
    assert list(Xs.columns) == list(X.columns)


def test_get_feature_matrix_drops_target(tiny_raw_df: pd.DataFrame) -> None:
    X, y = get_feature_matrix(tiny_raw_df)
    assert TARGET_COL not in X.columns
    assert y.name == TARGET_COL
    assert len(X) == len(y)
