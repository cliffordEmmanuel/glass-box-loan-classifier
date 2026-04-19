"""SHAP, LIME, DiCE, and partial-dependence explanation utilities.

Structure
---------
- Global explanations: SHAP beeswarm + bar, partial dependence plots.
- Local explanations: SHAP waterfall, LIME, adverse-action reason codes.
- Consistency: measure SHAP vs LIME top-k agreement; saved to JSON for the
  dashboard to load (no more hard-coded strings).
- Actionable recourse: DiCE counterfactuals using the ``genetic`` method
  (more principled than ``random``), with immutability constraints on age
  and past delinquency history.
"""

from __future__ import annotations

import json
import logging

import dice_ml
import joblib
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay

from src.config import EXPLAIN_OUTPUT_DIR, MODELS_DIR, RANDOM_STATE, configure_logging
from src.data_ingestion import TARGET_COL, load_and_split
from src.preprocessing import Preprocessor, get_feature_matrix
from src.shap_compat import shap_xgb_compat

logger = logging.getLogger(__name__)

IMMUTABLE_FEATURES = ["age"]
INCREASING_ONLY = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model_and_data(model_name: str = "xgboost"):
    """Load persisted model and prepare test data.

    Reuses the persisted ``Preprocessor`` so inference runs with the exact
    same statistics the model was trained on — no re-fitting on test data.
    """
    model = joblib.load(MODELS_DIR / f"{model_name}.joblib")
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")

    # Prefer the fitted preprocessor if present; else fit fresh (backward-compat).
    preproc_path = MODELS_DIR / "preprocessor.joblib"
    if preproc_path.exists():
        preproc: Preprocessor = joblib.load(preproc_path)
    else:
        logger.warning("preprocessor.joblib not found — fitting on train set (legacy path).")
        train_df, _ = load_and_split()
        preproc = Preprocessor().fit(train_df)

    _, test_df = load_and_split()
    test_clean = preproc.transform(test_df)
    X_test, y_test = get_feature_matrix(test_clean)

    return model, X_test, y_test, feature_names, preproc


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values for a tree model. Handles the XGBoost 2.x / SHAP bug
    via a tightly-scoped ``float`` shim — see ``src.shap_compat``."""
    # Unwrap CalibratedClassifierCV to get the underlying tree model for SHAP.
    base = _unwrap_calibrated(model)
    with shap_xgb_compat():
        explainer = shap.TreeExplainer(base)
    return explainer(X)


def _unwrap_calibrated(model):
    """Return the underlying tree model that SHAP's TreeExplainer can read.

    Peels two layers of sklearn wrapping when present:
    - ``CalibratedClassifierCV`` (the outer calibrator), whose
      ``calibrated_classifiers_[i].estimator`` holds the base estimator.
    - ``FrozenEstimator`` (used by sklearn >= 1.6 to replace ``cv='prefit'``),
      whose ``.estimator`` attribute holds the actual fitted model.
    """
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        # All calibrated_classifiers_ wrap the same base — use the first.
        base = model.calibrated_classifiers_[0].estimator
        # FrozenEstimator is a thin wrapper; unwrap it for SHAP.
        if hasattr(base, "estimator") and base.__class__.__name__ == "FrozenEstimator":
            base = base.estimator
        return base
    return model


def plot_shap_global(shap_values: shap.Explanation, X: pd.DataFrame, save: bool = True) -> None:
    EXPLAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Global Feature Importance (Beeswarm)")
    plt.tight_layout()
    if save:
        plt.savefig(EXPLAIN_OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", EXPLAIN_OUTPUT_DIR / "shap_summary.png")
    plt.close("all")

    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title("SHAP Mean Absolute Feature Importance")
    plt.tight_layout()
    if save:
        plt.savefig(EXPLAIN_OUTPUT_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", EXPLAIN_OUTPUT_DIR / "shap_bar.png")
    plt.close("all")


def plot_shap_local(shap_values: shap.Explanation, idx: int, save: bool = True) -> None:
    EXPLAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.title(f"SHAP Waterfall — Applicant #{idx}")
    plt.tight_layout()
    if save:
        plt.savefig(EXPLAIN_OUTPUT_DIR / f"shap_waterfall_{idx}.png", dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", EXPLAIN_OUTPUT_DIR / f"shap_waterfall_{idx}.png")
    plt.close("all")


def get_shap_feature_ranking(shap_values: shap.Explanation, idx: int) -> list[str]:
    abs_vals = np.abs(shap_values[idx].values)
    order = np.argsort(-abs_vals)
    return [shap_values.feature_names[i] for i in order]


# ---------------------------------------------------------------------------
# Partial dependence
# ---------------------------------------------------------------------------

def plot_partial_dependence(model, X: pd.DataFrame, features: list[str] | None = None,
                            save: bool = True) -> None:
    """Partial dependence for the most important features.

    PDP complements SHAP by showing the *average* shape of each feature's
    effect (e.g. monotonicity), which SHAP beeswarms can obscure when there
    are strong interactions.
    """
    EXPLAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features = features or [
        "RevolvingUtilizationOfUnsecuredLines",
        "DebtRatio",
        "MonthlyIncome",
        "age",
    ]
    features = [f for f in features if f in X.columns]
    if not features:
        logger.warning("No PDP features found in X.")
        return

    fig, axes = plt.subplots(
        nrows=(len(features) + 1) // 2, ncols=2,
        figsize=(12, 4 * ((len(features) + 1) // 2)),
    )
    PartialDependenceDisplay.from_estimator(
        model, X, features, ax=axes.ravel() if hasattr(axes, "ravel") else axes,
        n_jobs=-1, grid_resolution=30,
    )
    fig.suptitle("Partial Dependence — average feature effect on P(default)", fontsize=12)
    fig.tight_layout()
    if save:
        path = EXPLAIN_OUTPUT_DIR / "partial_dependence.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------

def create_lime_explainer(X_train: pd.DataFrame) -> lime.lime_tabular.LimeTabularExplainer:
    return lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        class_names=["No Default", "Default"],
        mode="classification",
        random_state=RANDOM_STATE,
    )


def explain_with_lime(lime_explainer, model, instance: np.ndarray, num_features: int = 10):
    return lime_explainer.explain_instance(
        instance, model.predict_proba, num_features=num_features,
    )


def get_lime_feature_ranking(lime_explanation) -> list[str]:
    feature_weights = lime_explanation.as_list()
    ranked = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)
    return [name for name, _ in ranked]


def plot_lime_local(lime_explanation, idx: int, save: bool = True) -> None:
    EXPLAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = lime_explanation.as_pyplot_figure()
    fig.suptitle(f"LIME Explanation — Applicant #{idx}", fontsize=12)
    plt.tight_layout()
    if save:
        fig.savefig(EXPLAIN_OUTPUT_DIR / f"lime_explanation_{idx}.png", dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", EXPLAIN_OUTPUT_DIR / f"lime_explanation_{idx}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# SHAP vs LIME consistency
# ---------------------------------------------------------------------------

def measure_consistency(
    shap_values: shap.Explanation,
    lime_explainer,
    model,
    X_test: pd.DataFrame,
    n_samples: int = 100,
    top_k: int = 5,
    save: bool = True,
) -> dict:
    """How often do SHAP and LIME agree on the top-k features? Saves JSON."""
    rng = np.random.RandomState(RANDOM_STATE)
    indices = rng.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)

    agreements = []
    for idx in indices:
        shap_rank = set(get_shap_feature_ranking(shap_values, idx)[:top_k])
        lime_exp = explain_with_lime(lime_explainer, model, X_test.iloc[idx].values)
        lime_rank_raw = get_lime_feature_ranking(lime_exp)
        lime_features = set()
        for feat_str in lime_rank_raw[:top_k]:
            for col in X_test.columns:
                if col in feat_str:
                    lime_features.add(col)
                    break
        agreements.append(len(shap_rank & lime_features) / top_k)

    result = {
        "n_samples": int(len(indices)),
        "top_k": int(top_k),
        "mean_agreement": round(float(np.mean(agreements)), 4),
        "std_agreement": round(float(np.std(agreements)), 4),
        "min_agreement": round(float(np.min(agreements)), 4),
        "max_agreement": round(float(np.max(agreements)), 4),
    }

    logger.info("SHAP vs LIME top-%d agreement (n=%d): %.1f%% ± %.1f%%",
                top_k, len(indices),
                result["mean_agreement"] * 100, result["std_agreement"] * 100)
    verdict = (
        "LOW — explanations may not be trustworthy"
        if result["mean_agreement"] < 0.6
        else "MODERATE — caution warranted"
        if result["mean_agreement"] < 0.8
        else "HIGH — explanations are consistent"
    )
    result["verdict"] = verdict
    logger.info("Consistency verdict: %s", verdict)

    if save:
        EXPLAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = EXPLAIN_OUTPUT_DIR / "consistency.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved: %s", path)

    return result


# ---------------------------------------------------------------------------
# DiCE counterfactuals
# ---------------------------------------------------------------------------

def create_dice_explainer(model, X_train: pd.DataFrame, method: str = "genetic") -> dice_ml.Dice:
    """Create a DiCE explainer.

    ``method="genetic"`` is more principled than ``"random"``: it searches for
    counterfactuals that are close to the query, diverse, and satisfy the
    declared feasibility constraints. It is slower but produces realistic
    recourse advice.
    """
    train_with_target = X_train.copy()
    train_with_target[TARGET_COL] = 0  # placeholder — DiCE needs the schema
    data = dice_ml.Data(
        dataframe=train_with_target,
        continuous_features=list(X_train.columns),
        outcome_name=TARGET_COL,
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    return dice_ml.Dice(data, dice_model, method=method)


def generate_counterfactuals(
    dice_explainer: dice_ml.Dice,
    instance: pd.DataFrame,
    total_cfs: int = 4,
    desired_class: int = 0,
):
    features_to_vary = [
        col for col in instance.columns
        if col not in IMMUTABLE_FEATURES and col not in INCREASING_ONLY
    ]
    permitted_range = {}
    for col in INCREASING_ONLY:
        current = float(instance[col].iloc[0])
        permitted_range[col] = [current, current]  # frozen at current value

    return dice_explainer.generate_counterfactuals(
        query_instances=instance,
        total_CFs=total_cfs,
        desired_class=desired_class,
        features_to_vary=features_to_vary,
        permitted_range=permitted_range,
    )


def format_counterfactual_advice(
    original: pd.Series,
    counterfactual: pd.Series,
    feature_names: list[str],
) -> list[str]:
    advice: list[str] = []
    for feat in feature_names:
        if feat in IMMUTABLE_FEATURES or feat in INCREASING_ONLY:
            continue
        orig_val = original[feat]
        cf_val = counterfactual[feat]
        diff = cf_val - orig_val
        if abs(diff) < 1e-6:
            continue
        direction = "Increase" if diff > 0 else "Decrease"

        if feat == "MonthlyIncome":
            advice.append(f"{direction} monthly income by ${abs(diff):,.0f} (to ${cf_val:,.0f})")
        elif feat == "DebtRatio":
            advice.append(f"{direction} debt ratio by {abs(diff):.2f} (to {cf_val:.2f})")
        elif feat == "RevolvingUtilizationOfUnsecuredLines":
            advice.append(f"{direction} credit utilisation by {abs(diff):.2f} (to {cf_val:.2f})")
        elif feat == "NumberOfOpenCreditLinesAndLoans":
            advice.append(f"{direction} open credit lines by {abs(diff):.0f} (to {cf_val:.0f})")
        elif feat == "NumberRealEstateLoansOrLines":
            advice.append(f"{direction} real estate loans by {abs(diff):.0f} (to {cf_val:.0f})")
        elif feat == "NumberOfDependents":
            advice.append(f"{direction} number of dependents by {abs(diff):.0f} (to {cf_val:.0f})")
        else:
            advice.append(f"{direction} {feat} by {abs(diff):.2f}")
    return advice


# ---------------------------------------------------------------------------
# Standalone pipeline entry point (Phase 3)
# ---------------------------------------------------------------------------

def run_explanations() -> None:
    logger.info("Loading model and data...")
    model, X_test, y_test, feature_names, preproc = load_model_and_data("xgboost")

    X_sample = X_test.iloc[:1000]

    logger.info("Computing SHAP values...")
    shap_values = compute_shap_values(model, X_sample)

    logger.info("Generating global SHAP plots...")
    plot_shap_global(shap_values, X_sample)

    logger.info("Generating partial dependence plots...")
    try:
        plot_partial_dependence(model, X_sample)
    except Exception as e:
        logger.warning("PDP skipped: %s", e)

    y_pred = model.predict(X_test)
    rejected_indices = np.where(y_pred == 1)[0]
    sample_idx = int(rejected_indices[0]) if len(rejected_indices) > 0 else 0
    sample_idx = min(sample_idx, len(X_sample) - 1)

    logger.info("Generating local SHAP for applicant #%d", sample_idx)
    plot_shap_local(shap_values, sample_idx)

    logger.info("Preparing LIME explainer...")
    train_df, _ = load_and_split()
    train_clean = preproc.transform(train_df)
    X_train, _ = get_feature_matrix(train_clean)
    lime_explainer = create_lime_explainer(X_train)

    logger.info("Generating LIME for applicant #%d", sample_idx)
    lime_exp = explain_with_lime(lime_explainer, model, X_sample.iloc[sample_idx].values)
    plot_lime_local(lime_exp, sample_idx)

    logger.info("Running SHAP vs LIME consistency check...")
    measure_consistency(shap_values, lime_explainer, model, X_sample, n_samples=50)

    logger.info("Generating DiCE counterfactual for applicant #%d", sample_idx)
    try:
        dice_explainer = create_dice_explainer(model, X_train)
        query = X_sample.iloc[[sample_idx]]
        cfs = generate_counterfactuals(dice_explainer, query, total_cfs=2)
        cf_df = cfs.cf_examples_list[0].final_cfs_df
        if cf_df is not None and len(cf_df) > 0:
            logger.info("DiCE produced %d counterfactual(s).", len(cf_df))
        else:
            logger.warning("DiCE produced no valid counterfactuals for this applicant.")
    except Exception as e:
        logger.warning("DiCE skipped: %s", e)

    logger.info("Phase 3 complete.")


if __name__ == "__main__":
    configure_logging()
    run_explanations()
