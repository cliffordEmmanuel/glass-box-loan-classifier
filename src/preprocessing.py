"""Scaling, encoding, and imputation logic.

The pipeline is encapsulated in a ``Preprocessor`` object that follows the
scikit-learn fit/transform contract. **All statistics (medians, scaler means
and stds) are learned on the training split and applied unchanged to any
other split.** This is the correctness-critical part of a credit model — if
test-set medians leaked into the pipeline, our holdout metrics and our
explanations would both be optimistic.

Imputation
----------
Median imputation for ``MonthlyIncome`` and ``NumberOfDependents``.

Why median (not mean)?
- ``MonthlyIncome`` is right-skewed (high earners pull the mean up). The mean
  would systematically overstate income for records with missing values,
  which could disproportionately favour higher-income demographic groups at
  decisioning time.
- Median is robust to outliers and preserves the central tendency for the
  majority of borrowers.

Why not model-based imputation?
- For a "glass box" project, deterministic median imputation is transparent
  and reproducible. Fancy methods (MICE, KNN) would add a hidden modelling
  step that is harder to audit.

Outlier handling
----------------
``RevolvingUtilizationOfUnsecuredLines`` and ``DebtRatio`` contain extreme
values (some > 1e5). We cap these at domain thresholds (1.5 and 5.0
respectively) to prevent them from dominating the model.

Past-due sentinels
------------------
The three past-due columns encode 96 and 98 as sentinel "unknown" values in
the original Kaggle data dictionary. We remap them to the *training* median
of the non-sentinel values in the same column.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_ingestion import TARGET_COL

COLS_WITH_MISSING = ["MonthlyIncome", "NumberOfDependents"]

OUTLIER_CAPS = {
    "RevolvingUtilizationOfUnsecuredLines": 1.5,
    "DebtRatio": 5.0,
}

PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
PAST_DUE_SENTINELS = (96, 98)


@dataclass
class Preprocessor:
    """Stateful preprocessor that learns statistics on train and applies them
    to any subsequent split.

    The fit/transform pattern is deliberate: it is the mechanism that prevents
    test-set leakage. Use :meth:`fit_transform` on the training dataframe and
    :meth:`transform` on validation/test/inference dataframes.
    """

    impute_medians_: dict[str, float] = field(default_factory=dict)
    past_due_medians_: dict[str, float] = field(default_factory=dict)
    fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        """Learn imputation medians from the training set only."""
        self.impute_medians_ = {col: float(df[col].median()) for col in COLS_WITH_MISSING if col in df.columns}

        self.past_due_medians_ = {}
        for col in PAST_DUE_COLS:
            if col not in df.columns:
                continue
            mask = df[col].isin(PAST_DUE_SENTINELS)
            non_sentinel = df.loc[~mask, col]
            self.past_due_medians_[col] = float(non_sentinel.median())

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned statistics. Safe to call on any split."""
        if not self.fitted_:
            raise RuntimeError("Preprocessor must be fit before transform.")
        out = df.copy()

        for col, median in self.impute_medians_.items():
            if col in out.columns:
                out[col] = out[col].fillna(median)

        for col, cap in OUTLIER_CAPS.items():
            if col in out.columns:
                out[col] = out[col].clip(upper=cap)

        for col, median in self.past_due_medians_.items():
            if col in out.columns:
                mask = out[col].isin(PAST_DUE_SENTINELS)
                if mask.any():
                    out.loc[mask, col] = median

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a preprocessed dataframe into X (features) and y (target)."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


@dataclass
class FeatureScaler:
    """Thin wrapper around ``StandardScaler`` that preserves DataFrame semantics
    and mirrors the Preprocessor fit/transform contract for clarity."""

    scaler_: StandardScaler | None = None
    columns_: list[str] | None = None

    def fit(self, X: pd.DataFrame) -> FeatureScaler:
        self.scaler_ = StandardScaler().fit(X)
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_ is None or self.columns_ is None:
            raise RuntimeError("FeatureScaler must be fit before transform.")
        X = X[self.columns_]  # enforce column order
        return pd.DataFrame(
            self.scaler_.transform(X),
            columns=self.columns_,
            index=X.index,
        )

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


# Back-compat shims: the old module-level functions are still imported by
# notebooks and old scripts. We keep them but make sure they run in a
# no-leakage way by fitting a local Preprocessor on the given df. Prefer the
# Preprocessor class in new code.

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Deprecated: use ``Preprocessor().fit_transform(df)`` for train and
    ``Preprocessor.transform(df)`` with a pre-fit preprocessor for test."""
    return Preprocessor().fit_transform(df)


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Deprecated: use ``FeatureScaler`` instead. Fits on train, transforms both."""
    fs = FeatureScaler().fit(X_train)
    return fs.transform(X_train), fs.transform(X_test), fs.scaler_
