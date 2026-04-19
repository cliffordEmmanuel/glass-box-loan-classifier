"""Population Stability Index (PSI) drift detection.

PSI is the standard measure in credit risk for checking whether the
distribution of a feature has shifted between two samples (usually the
training population and live scoring traffic). For credit models a common
rule of thumb is:

    PSI < 0.10   : no material shift
    0.10 ≤ PSI < 0.25 : moderate shift — investigate
    PSI ≥ 0.25   : major shift — the model may no longer be valid

We compute PSI with quantile-based bucketing of the *reference* distribution
so the metric is well-defined even for skewed features. Zero-frequency
buckets are smoothed with a small epsilon to avoid ``-inf`` terms.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DRIFT_OUTPUT_DIR

logger = logging.getLogger(__name__)

_EPS = 1e-6


def _bucketise(reference: np.ndarray, n_buckets: int) -> np.ndarray:
    """Return bucket edges from the reference distribution (quantile-based).

    Duplicate edges are collapsed so features with mass points (e.g. many
    zeros in a past-due count) degrade gracefully to fewer buckets.
    """
    quantiles = np.linspace(0, 1, n_buckets + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    # Pad the ends so any out-of-range value in the new distribution still
    # lands in a bucket.
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def psi_for_feature(
    reference: np.ndarray,
    current: np.ndarray,
    n_buckets: int = 10,
) -> float:
    """Population Stability Index for a single numeric feature."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    edges = _bucketise(reference, n_buckets)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()
    ref_pct = np.where(ref_pct == 0, _EPS, ref_pct)
    cur_pct = np.where(cur_pct == 0, _EPS, cur_pct)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def psi_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Compute PSI per feature and return a ranked DataFrame."""
    common = [c for c in reference_df.columns if c in current_df.columns]
    rows = []
    for col in common:
        psi = psi_for_feature(reference_df[col].values, current_df[col].values, n_buckets)
        if psi < 0.10:
            severity = "OK"
        elif psi < 0.25:
            severity = "MODERATE"
        else:
            severity = "MAJOR"
        rows.append({"feature": col, "psi": round(psi, 4), "severity": severity})

    report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return report


def save_psi_report(report: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Persist the PSI report as CSV + JSON for dashboard consumption."""
    output_dir = output_dir or DRIFT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "psi_report.csv"
    report.to_csv(csv_path, index=False)
    with open(output_dir / "psi_report.json", "w") as f:
        json.dump(report.to_dict(orient="records"), f, indent=2)
    logger.info("PSI report saved to %s", csv_path)
    return csv_path


def run_drift_check() -> pd.DataFrame:
    """Standalone entry point: compare train vs test split as a sanity check.

    In production you would compare *training* data against a recent window
    of live scoring traffic. For the portfolio project, comparing train and
    test gives us a baseline PSI (which should be close to zero for a random
    stratified split — any non-trivial value flags a bug).
    """
    from src.data_ingestion import TARGET_COL, load_and_split

    train_df, test_df = load_and_split()
    features = [c for c in train_df.columns if c != TARGET_COL]
    report = psi_report(train_df[features], test_df[features])
    save_psi_report(report)
    logger.info("PSI report (train vs test):\n%s", report.to_string(index=False))
    return report


if __name__ == "__main__":
    from src.config import configure_logging

    configure_logging()
    run_drift_check()
