# Glass Box Loan Classifier

Explainable AI (XAI) dashboard for credit scoring models.

## Project Overview

Finance models are often "black boxes," making it hard to explain to a customer why their loan was denied. This tool provides transparency through SHAP and LIME explanations, DiCE counterfactuals, adverse-action-style reason codes, partial-dependence plots, isotonic probability calibration, a cost-aware decision threshold, and a PSI drift monitor — all surfaced through an interactive dashboard.

Built on the Kaggle [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset (150,000 borrower records), the project trains three interpretable and black-box models side by side, tunes the black-box with 5-fold CV grid search, calibrates its probabilities, and layers on explainability tools so a loan officer can understand — and act on — every decision.

See [`docs/model_card.md`](docs/model_card.md) for intended use, training-data details, fairness considerations, and known limitations.

## Key Features

- **Model Duel**: Logistic Regression and Decision Tree (glass box) vs. tuned XGBoost (black box), compared with 5-fold CV ROC-AUC plus held-out test metrics. Any of the three is selectable in the dashboard.
- **Honest probabilities**: XGBoost is wrapped in `CalibratedClassifierCV` (isotonic) so "Default risk: 37%" actually means 37%.
- **Cost-aware threshold**: Training searches thresholds to minimise expected loss under a configurable false-approve / false-deny cost ratio. The dashboard exposes a slider.
- **Global explanations**: SHAP beeswarm and bar plots plus partial-dependence plots showing the average shape of each feature's effect.
- **Local explanations**: SHAP waterfall, LIME, and plain-English adverse-action reason codes for every denied applicant.
- **Consistency check**: SHAP vs. LIME top-5 agreement, computed on sampled test instances and saved to JSON for the dashboard.
- **Actionable recourse**: DiCE counterfactuals (`genetic` method) with immutability constraints — age is frozen, past delinquencies can never decrease.
- **Drift monitor**: Population Stability Index (PSI) per feature, with OK / MODERATE / MAJOR severity flags.
- **Run log**: Each training run appends hyperparameters, CV metrics, and tuned threshold to `outputs/models/run_log.csv`.

## Stack

- Python 3.11+
- Scikit-learn / XGBoost
- SHAP / LIME / DiCE
- Streamlit (dashboard)
- pytest / ruff / mypy
- uv (dependency management)

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Homebrew `libomp` on macOS (required by XGBoost): `brew install libomp`

## Quick Start

```bash
# 1. Install dependencies (including dev tools)
make install

# 2. Download the dataset (one-time)
make data

# 3. Run the full pipeline (EDA → training → explanations → drift)
make pipeline

# 4. Launch the dashboard
make dashboard
```

## Individual Phases

```bash
make eda        # Phase 1 — EDA (writes PNGs to outputs/eda/)
make train      # Phase 2 — Train, tune, calibrate, threshold (writes models/ + run_log.csv)
make explain    # Phase 3 — SHAP, LIME, PDP, DiCE, consistency.json
make drift      # Phase 4 — PSI train vs test → outputs/drift/
make dashboard  # Phase 5 — Streamlit
```

Or without Make:

```bash
uv run python -m src.eda
uv run python -m src.model_trainer
uv run python -m src.xai_engine
uv run python -m src.drift
uv run streamlit run app/main_ui.py
```

## Testing and Quality Gates

```bash
make test        # pytest
make lint        # ruff
make typecheck   # mypy (non-blocking)
make fmt         # ruff check --fix + ruff format
```

CI runs `ruff` and `pytest` on every push / PR (see `.github/workflows/ci.yml`).

## Configuration

Override any path or cost parameter via environment variables:

```bash
export GBLC_COST_FA=3.0      # cost of a false approve (relative)
export GBLC_COST_FD=1.0      # cost of a false deny
export GBLC_THRESHOLD=0.4    # fallback threshold when no training run has been done
export GBLC_DATA_DIR=/path/to/data
export GBLC_MODELS_DIR=/path/to/models
```

Defaults live in `src/config.py`.

## Directory Structure

```
├── app/
│   └── main_ui.py              # Streamlit dashboard
├── data/
│   ├── raw/                    # Downloaded dataset
│   └── processed/              # Train/test splits (cached)
├── docs/
│   └── model_card.md           # Model card (intended use, fairness, limits)
├── models/                     # Trained .joblib artifacts + threshold.json
├── outputs/
│   ├── eda/                    # EDA plots
│   ├── explanations/           # SHAP, LIME, PDP, consistency.json
│   ├── models/                 # Model comparison CSV/JSON + run_log.csv
│   └── drift/                  # PSI report
├── src/
│   ├── config.py               # Paths, thresholds, costs, logging setup
│   ├── data_ingestion.py       # Load + split
│   ├── preprocessing.py        # Preprocessor class (fit/transform, no leakage)
│   ├── eda.py                  # EDA — writes PNGs to outputs/eda/
│   ├── model_trainer.py        # Train, tune, calibrate, threshold, run log
│   ├── xai_engine.py           # SHAP, LIME, PDP, DiCE (genetic), consistency
│   ├── reason_codes.py         # Adverse-action reason-code generator
│   ├── drift.py                # PSI
│   └── shap_compat.py          # Scoped SHAP/XGBoost float shim
├── tests/                      # pytest tests (preprocessing, drift, reasons, threshold, shap_compat)
├── .github/workflows/ci.yml    # Lint + test on push/PR
├── Makefile                    # Common recipes
├── main.py                     # Full pipeline entry point
└── pyproject.toml              # Project config, pinned deps, ruff/mypy config
```

## What's New in 0.2

- **No more data leakage.** Preprocessing is now a `Preprocessor` class with a `fit/transform` contract; train medians are used to transform test.
- **Probability calibration** via `CalibratedClassifierCV` so displayed percentages are meaningful.
- **Cost-aware threshold**: tuned on the training set and exposed via a dashboard slider.
- **Model selector** in the dashboard: any of the three candidate models.
- **Adverse-action reason codes** for denied applicants.
- **Partial dependence plots** alongside SHAP.
- **PSI drift monitor** as a fourth pipeline phase.
- **Consistency score** now read from JSON, not hard-coded in the UI.
- **SHAP monkey-patch** replaced with a tightly-scoped context manager (`src/shap_compat.py`).
- **Hyperparameter tuning** (modest grid search) and **stratified 5-fold CV** for model selection.
- **Tests, linting, CI, Makefile, model card, run log.**

### Latest training-run metrics

| model                | threshold | accuracy | F1    | ROC-AUC | CV AUC (mean ± std) |
|----------------------|-----------|----------|-------|---------|---------------------|
| Logistic Regression  | 0.50      | 0.800    | 0.335 | 0.859   | 0.853 ± 0.004       |
| Decision Tree        | 0.50      | 0.721    | 0.283 | 0.846   | 0.840 ± 0.003       |
| XGBoost (raw)        | 0.50      | 0.792    | 0.335 | 0.868   | 0.864 ± 0.003       |
| **XGBoost (calibrated)** | **0.16 (tuned)** | **0.898** | **0.436** | **0.867** | 0.864 ± 0.003 |

Calibration plus the cost-aware threshold lifts F1 from 0.34 to 0.44 while preserving AUC. PSI train-vs-test shows all features below 0.001 (no drift, as expected for a random split).

## Dashboard Usage

1. **Single Applicant tab** — choose a model from the sidebar, slide the threshold (initialised from the tuned value in `models/threshold.json`), enter applicant details (or load an example — approved or denied), press **Evaluate Application**. Inputs are validated (e.g. 90-day-late count cannot exceed 30-day-late count). Toggle SHAP, LIME, and DiCE counterfactuals; denied applicants also get plain-English adverse-action reason codes.
2. **Model Insights tab** — global SHAP feature importance, partial-dependence plots, model performance comparison, SHAP/LIME consistency score (loaded from `outputs/explanations/consistency.json`).
3. **Data Explorer tab** — correlation matrix (proxy-variable check), target correlation, feature distributions.
4. **Drift Monitor tab** — PSI report per feature with OK / MODERATE / MAJOR severity flags.

## Design Notes

**Why EDA lives in a `.py` file, not a notebook.** The project's goal is auditability and reproducibility, which favours scripts. `src/eda.py` runs deterministically under `make eda`, writes PNGs to `outputs/eda/`, diffs cleanly in git, is covered by the same lint/type gates as the rest of the codebase, and exposes functions (`plot_correlation_matrix`, `plot_feature_distributions`, …) that can be imported from other code. If a narrative walkthrough is needed for a stakeholder, the cleanest extension is a thin `notebooks/eda.ipynb` that imports from `src.eda` rather than duplicating the logic.

**Why a `Preprocessor` class instead of loose functions.** The class enforces the fit/transform contract that prevents test-set statistics from leaking into training — medians are learned on the train split and applied unchanged to test. A regression test (`tests/test_preprocessing.py::test_preprocessor_no_leakage_between_fit_and_transform`) pins this behaviour.

**Why calibration is a separate step.** Raw XGBoost margins don't line up with real default frequencies, so "37% risk" wouldn't mean what it says. Isotonic calibration on a held-out split aligns them, and SHAP explanations are still extracted from the underlying booster via a small `_unwrap_calibrated()` helper.
