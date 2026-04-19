"""Central configuration — paths, thresholds, and cost parameters.

Keeping these in one module means the dashboard, training, and explanation
pipelines share a single source of truth. Environment variables can override
the defaults for non-interactive runs (CI, scheduled retraining, etc.).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("GBLC_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = Path(os.environ.get("GBLC_MODELS_DIR", PROJECT_ROOT / "models"))
OUTPUTS_DIR = Path(os.environ.get("GBLC_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"
MODEL_OUTPUT_DIR = OUTPUTS_DIR / "models"
EXPLAIN_OUTPUT_DIR = OUTPUTS_DIR / "explanations"
DRIFT_OUTPUT_DIR = OUTPUTS_DIR / "drift"
DOCS_DIR = PROJECT_ROOT / "docs"

# --- Decision threshold ------------------------------------------------------
#
# 0.5 is almost never the business-optimal cutoff for a class-imbalanced
# credit model. The cost of a false approve (lending to someone who defaults)
# dwarfs the cost of a false deny (foregone interest). The default below is
# tuned to roughly minimise expected loss under the cost ratio declared here;
# the dashboard lets an operator override it.
#
COST_FALSE_APPROVE = float(os.environ.get("GBLC_COST_FA", 5.0))   # relative cost
COST_FALSE_DENY = float(os.environ.get("GBLC_COST_FD", 1.0))
DEFAULT_THRESHOLD = float(os.environ.get("GBLC_THRESHOLD", 0.35))

RANDOM_STATE = 42


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a uniform log format for the pipeline scripts.

    Called from ``main.py`` and from each ``__main__`` block. Safe to call
    multiple times — it only adds handlers if none exist.
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)
