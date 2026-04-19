"""Load and split the Give Me Some Credit dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_DIR, RANDOM_STATE, RAW_DATA_DIR

logger = logging.getLogger(__name__)

TARGET_COL = "SeriousDlqin2yrs"

FEATURE_DESCRIPTIONS = {
    "RevolvingUtilizationOfUnsecuredLines": "Total balance on credit cards and personal lines of credit / sum of credit limits",
    "age": "Age of borrower in years",
    "NumberOfTime30-59DaysPastDueNotWorse": "Number of times borrower has been 30-59 days past due (not worse) in the last 2 years",
    "DebtRatio": "Monthly debt payments, alimony, living costs / monthly gross income",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Number of open loans and lines of credit",
    "NumberOfTimes90DaysLate": "Number of times borrower has been 90+ days late",
    "NumberRealEstateLoansOrLines": "Number of mortgage and real estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "Number of times borrower has been 60-89 days past due (not worse) in the last 2 years",
    "NumberOfDependents": "Number of dependents in family (excluding borrower)",
}


def load_raw_data(filename: str = "cs-training.csv", data_dir: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV dataset, dropping the unnamed index column."""
    filepath = (data_dir or RAW_DATA_DIR) / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {filepath}. "
            "See README 'Quick Start' for the curl command to download it."
        )
    df = pd.read_csv(filepath)
    if df.columns[0] == "" or df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split preserving class balance."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Persist train/test splits to the processed data directory."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)


def load_and_split(force_reload: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: load raw data, split, save, and return splits.

    If the processed files already exist, reuse them (so downstream scripts
    share the same holdout) unless ``force_reload`` is set.
    """
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    if not force_reload and train_path.exists() and test_path.exists():
        logger.debug("Reusing cached splits at %s", PROCESSED_DATA_DIR)
        return pd.read_csv(train_path), pd.read_csv(test_path)

    df = load_raw_data()
    train_df, test_df = split_data(df)
    save_splits(train_df, test_df)
    logger.info("Saved splits: %d train / %d test", len(train_df), len(test_df))
    return train_df, test_df
