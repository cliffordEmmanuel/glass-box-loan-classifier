"""Glass Box Loan Classifier — project entry point.

Usage
-----
    # Full pipeline (EDA, training, explanations, drift check):
    uv run python main.py

    # Individual phases:
    uv run python -m src.eda              # Phase 1: EDA
    uv run python -m src.model_trainer    # Phase 2: Train models
    uv run python -m src.xai_engine       # Phase 3: Generate explanations
    uv run python -m src.drift            # Phase 4: PSI drift check

    # Dashboard:
    uv run streamlit run app/main_ui.py   # Phase 5: Dashboard
"""

from __future__ import annotations

import logging

from src.config import configure_logging
from src.drift import run_drift_check
from src.eda import run_eda
from src.model_trainer import train_and_compare
from src.xai_engine import run_explanations

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    logger.info("=" * 60)
    logger.info("  Glass Box Loan Classifier — Full Pipeline")
    logger.info("=" * 60)

    logger.info("[Phase 1] Exploratory Data Analysis")
    run_eda()

    logger.info("[Phase 2] Model Training & Comparison")
    train_and_compare()

    logger.info("[Phase 3] Explainability (SHAP, LIME, PDP, DiCE)")
    run_explanations()

    logger.info("[Phase 4] Drift Check (PSI, train vs test)")
    run_drift_check()

    logger.info("=" * 60)
    logger.info("  Pipeline complete.")
    logger.info("  Launch the dashboard with:")
    logger.info("    uv run streamlit run app/main_ui.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
