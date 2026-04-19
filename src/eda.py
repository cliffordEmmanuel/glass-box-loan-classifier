"""Exploratory Data Analysis for the Give Me Some Credit dataset.

Produces:
1. Dataset summary statistics and missing-value report
2. Class balance analysis
3. Correlation matrix heatmap (proxy variable check)
4. Feature distributions by target class
5. Feature correlation with the target
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import EDA_OUTPUT_DIR, configure_logging
from src.data_ingestion import TARGET_COL, load_raw_data

logger = logging.getLogger(__name__)


def generate_summary_report(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null": df.count(),
        "null_count": df.isna().sum(),
        "null_pct": (df.isna().sum() / len(df) * 100).round(2),
        "mean": df.mean(numeric_only=True),
        "median": df.median(numeric_only=True),
        "std": df.std(numeric_only=True),
        "min": df.min(numeric_only=True),
        "max": df.max(numeric_only=True),
    })
    logger.info("Dataset shape: %d rows x %d columns", df.shape[0], df.shape[1])
    logger.info("Target distribution (%s):", TARGET_COL)
    vc = df[TARGET_COL].value_counts()
    for val, count in vc.items():
        logger.info("  %s: %s (%.1f%%)", val, f"{count:,}", count / len(df) * 100)
    missing = summary[summary["null_count"] > 0][["null_count", "null_pct"]]
    if missing.empty:
        logger.info("Missing values: none")
    else:
        logger.info("Missing values:")
        for col, row in missing.iterrows():
            logger.info("  %s: %d (%.1f%%)", col, int(row["null_count"]), row["null_pct"])
    return summary


def plot_correlation_matrix(df: pd.DataFrame, save: bool = True) -> None:
    """Correlation heatmap — a first-pass proxy-variable check.

    If two features are highly correlated, one may serve as an indirect proxy
    for a protected attribute in the real-world population (e.g. zip code
    proxying for race). This dataset contains no protected attributes, so
    the check here is structural rather than protected-class fairness.
    """
    corr = df.drop(columns=[TARGET_COL]).corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Matrix\n(Proxy Variable Check)", fontsize=14)
    plt.tight_layout()

    if save:
        EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(EDA_OUTPUT_DIR / "correlation_matrix.png", dpi=150)
        logger.info("Saved: %s", EDA_OUTPUT_DIR / "correlation_matrix.png")
    plt.close(fig)

    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                high_corr.append((corr.columns[i], corr.columns[j], r))
    if high_corr:
        logger.info("Highly correlated feature pairs (|r| > 0.5):")
        for col1, col2, r in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
            logger.info("  %s <-> %s: r = %.3f", col1, col2, r)
    else:
        logger.info("No |r| > 0.5 pairs found — low structural proxy risk.")


def plot_feature_distributions(df: pd.DataFrame, save: bool = True) -> None:
    features = [c for c in df.columns if c != TARGET_COL]
    n_features = len(features)
    ncols = 3
    nrows = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for idx, col in enumerate(features):
        ax = axes[idx]
        for label in [0, 1]:
            subset = df[df[TARGET_COL] == label][col].dropna()
            q99 = subset.quantile(0.99)
            subset_clipped = subset.clip(upper=q99)
            ax.hist(subset_clipped, bins=50, alpha=0.5, label=f"Class {label}", density=True)
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)

    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Feature Distributions by Target Class", fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(EDA_OUTPUT_DIR / "feature_distributions.png", dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", EDA_OUTPUT_DIR / "feature_distributions.png")
    plt.close(fig)


def plot_target_correlation(df: pd.DataFrame, save: bool = True) -> None:
    corr_with_target = df.corr()[TARGET_COL].drop(TARGET_COL).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in corr_with_target.to_numpy()]
    corr_with_target.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Pearson Correlation with Target")
    ax.set_title("Feature Correlation with Loan Default (SeriousDlqin2yrs)")
    ax.axvline(x=0, color="black", linewidth=0.5)
    plt.tight_layout()

    if save:
        EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(EDA_OUTPUT_DIR / "target_correlation.png", dpi=150)
        logger.info("Saved: %s", EDA_OUTPUT_DIR / "target_correlation.png")
    plt.close(fig)


def run_eda() -> None:
    logger.info("Loading raw data...")
    df = load_raw_data()

    logger.info("--- Summary Report ---")
    generate_summary_report(df)

    logger.info("--- Correlation Matrix ---")
    plot_correlation_matrix(df)

    logger.info("--- Feature Distributions ---")
    plot_feature_distributions(df)

    logger.info("--- Target Correlation ---")
    plot_target_correlation(df)

    logger.info("EDA complete. Outputs saved to %s", EDA_OUTPUT_DIR)


if __name__ == "__main__":
    configure_logging()
    run_eda()
