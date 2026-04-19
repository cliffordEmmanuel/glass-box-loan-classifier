"""Train, tune, calibrate, and compare candidate models.

Key decisions
-------------
- **Three candidates.** Logistic Regression and Decision Tree (glass-box
  baselines) versus XGBoost (black-box challenger). The dashboard can serve
  any of the three — so the "Model Duel" is real, not just a bake-off.
- **Cross-validation.** Stratified 5-fold ROC-AUC on the training split is
  the selection metric. The held-out test split is only used for a single
  final evaluation per model.
- **Hyperparameter search.** XGBoost gets a small grid search (n_estimators,
  max_depth, learning_rate) using the same CV splits. LR and DT are small
  enough to tune via class-weight only. Grid is deliberately modest so a
  full run finishes in a few minutes on a laptop.
- **Calibration.** XGBoost with ``scale_pos_weight`` returns *ranked* but
  *uncalibrated* probabilities — good for AUC, misleading for the
  "Default risk: 37%" label in the UI. We wrap the best estimator in
  ``CalibratedClassifierCV`` (isotonic) so predicted probabilities mean
  what they say.
- **Cost-aware threshold.** Under the configured cost ratio
  (``COST_FALSE_APPROVE`` / ``COST_FALSE_DENY``) we search thresholds on
  the training folds and save the expected-loss-minimising value alongside
  the model artifacts. The dashboard defaults to this, with a slider.
- **Run log.** Every training run appends one row to
  ``outputs/models/run_log.csv`` (timestamp, params, CV mean/std, test AUC).
  A lightweight alternative to MLflow for a portfolio project.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

try:
    # sklearn >= 1.6: the supported way to calibrate a pre-trained model.
    from sklearn.frozen import FrozenEstimator as _FrozenEstimator
    _HAS_FROZEN = True
except ImportError:  # sklearn < 1.6 — still supports cv='prefit'
    _FrozenEstimator = None  # type: ignore[misc,assignment]
    _HAS_FROZEN = False
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    COST_FALSE_APPROVE,
    COST_FALSE_DENY,
    DEFAULT_THRESHOLD,
    MODEL_OUTPUT_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    configure_logging,
)
from src.data_ingestion import load_and_split
from src.preprocessing import FeatureScaler, Preprocessor, get_feature_matrix

logger = logging.getLogger(__name__)

N_CV_FOLDS = 5

XGB_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [150, 300],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
}


@dataclass
class TrainingArtifacts:
    xgb_model: Any
    xgb_calibrated: Any
    lr_model: LogisticRegression
    dt_model: DecisionTreeClassifier
    preprocessor: Preprocessor
    scaler: FeatureScaler
    feature_names: list[str]
    results: pd.DataFrame
    tuned_threshold: float
    cv_results: dict[str, dict[str, float]]


def _class_weight_ratio(y: pd.Series) -> float:
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    return n_neg / max(n_pos, 1)


def _cv_auc(estimator, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    """Return mean and std of ROC-AUC across stratified CV folds."""
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(estimator, X, y, cv=skf, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


def train_logistic_regression(X: pd.DataFrame, y: pd.Series, pos_weight: float) -> LogisticRegression:
    model = LogisticRegression(
        class_weight={0: 1.0, 1: pos_weight},
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    return model


def train_decision_tree(X: pd.DataFrame, y: pd.Series, pos_weight: float) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(
        class_weight={0: 1.0, 1: pos_weight},
        max_depth=5,
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    return model


def tune_xgboost(X: pd.DataFrame, y: pd.Series, pos_weight: float) -> tuple[XGBClassifier, dict[str, Any]]:
    """Modest grid search for XGBoost."""
    base = XGBClassifier(
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=base,
        param_grid=XGB_PARAM_GRID,
        scoring="roc_auc",
        cv=skf,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    logger.info("XGBoost grid search (%d combos x %d folds)...",
                np.prod([len(v) for v in XGB_PARAM_GRID.values()]), N_CV_FOLDS)
    grid.fit(X, y)
    logger.info("Best XGBoost params: %s (CV AUC = %.4f)", grid.best_params_, grid.best_score_)
    return grid.best_estimator_, grid.best_params_


def calibrate(model, X: pd.DataFrame, y: pd.Series) -> CalibratedClassifierCV:
    """Wrap a fitted classifier with isotonic probability calibration.

    We reuse the model we already trained on the full training set rather than
    holding out a separate calibration split. For a portfolio project this
    trade-off keeps more data for the base estimator; a production pipeline
    should carve out a real calibration split.

    sklearn 1.6 removed ``cv='prefit'`` and replaced it with the
    ``FrozenEstimator`` wrapper, which makes the "do not re-fit" intent
    explicit. We use whichever API the installed sklearn supports so this
    project works across 1.4 → 1.6+.
    """
    if _HAS_FROZEN:
        # cv=5 here only controls the isotonic calibrator's cross-fit folds;
        # the wrapped estimator is frozen and will not be re-trained.
        calibrator = CalibratedClassifierCV(
            _FrozenEstimator(model), method="isotonic", cv=5,
        )
    else:
        calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrator.fit(X, y)
    return calibrator


def find_cost_optimal_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    cost_fa: float = COST_FALSE_APPROVE,
    cost_fd: float = COST_FALSE_DENY,
) -> float:
    """Pick the threshold that minimises expected loss under the cost ratio.

    A false approve is a positive case (``y_true==1``) predicted 0; its cost
    is ``cost_fa``. A false deny is a negative case predicted 1; its cost is
    ``cost_fd``. We scan thresholds in [0.05, 0.95] at 0.01 resolution.
    """
    y_true_arr = np.asarray(y_true)
    best_threshold = 0.5
    best_cost = float("inf")
    for t in np.arange(0.05, 0.95, 0.01):
        pred = (y_prob >= t).astype(int)
        fa = int(((pred == 0) & (y_true_arr == 1)).sum())  # missed defaulters
        fd = int(((pred == 1) & (y_true_arr == 0)).sum())  # rejected good payers
        cost = fa * cost_fa + fd * cost_fd
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(t)
    logger.info("Cost-optimal threshold: %.2f (FA cost=%s, FD cost=%s)",
                best_threshold, cost_fa, cost_fd)
    return best_threshold


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, name: str, threshold: float = 0.5) -> dict[str, Any]:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "model": name,
        "threshold": round(threshold, 3),
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "f1_score": round(f1_score(y, y_pred), 4),
        "roc_auc": round(roc_auc_score(y, y_prob), 4),
    }
    cm = confusion_matrix(y, y_pred).tolist()

    logger.info("=" * 50)
    logger.info("  %s (threshold=%.2f)", name, threshold)
    logger.info("=" * 50)
    logger.info("  Accuracy:  %s", metrics["accuracy"])
    logger.info("  F1-Score:  %s", metrics["f1_score"])
    logger.info("  ROC-AUC:   %s", metrics["roc_auc"])
    logger.info("\n%s",
                classification_report(y, y_pred, target_names=["No Default", "Default"]))
    metrics["confusion_matrix"] = cm
    return metrics


def _append_run_log(output_dir: Path, row: dict[str, Any]) -> None:
    """Append a single-row CSV record for this run (poor-man's MLflow)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_log.csv"
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def train_and_compare(do_tune: bool = True) -> TrainingArtifacts:
    train_df, test_df = load_and_split()

    # Fit preprocessor on TRAIN ONLY; transform both splits with it.
    preproc = Preprocessor().fit(train_df)
    train_clean = preproc.transform(train_df)
    test_clean = preproc.transform(test_df)

    X_train, y_train = get_feature_matrix(train_clean)
    X_test, y_test = get_feature_matrix(test_clean)

    # Scale for models that need it (fit on train only).
    scaler = FeatureScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pos_weight = _class_weight_ratio(y_train)
    logger.info("Class weight ratio (neg/pos): %.2f", pos_weight)

    # --- Train candidates ---
    logger.info("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train_scaled, y_train, pos_weight)
    lr_cv_mean, lr_cv_std = _cv_auc(
        LogisticRegression(class_weight={0: 1.0, 1: pos_weight}, max_iter=1000,
                           solver="lbfgs", random_state=RANDOM_STATE),
        X_train_scaled, y_train,
    )

    logger.info("Training Decision Tree...")
    dt_model = train_decision_tree(X_train, y_train, pos_weight)
    dt_cv_mean, dt_cv_std = _cv_auc(
        DecisionTreeClassifier(class_weight={0: 1.0, 1: pos_weight}, max_depth=5,
                               random_state=RANDOM_STATE),
        X_train, y_train,
    )

    if do_tune:
        xgb_model, xgb_params = tune_xgboost(X_train, y_train, pos_weight)
    else:
        xgb_params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}
        xgb_model = XGBClassifier(
            scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=RANDOM_STATE, **xgb_params,
        ).fit(X_train, y_train)
    xgb_cv_mean, xgb_cv_std = _cv_auc(
        XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss",
                      random_state=RANDOM_STATE, **xgb_params),
        X_train, y_train,
    )

    cv_results = {
        "Logistic Regression": {"cv_auc_mean": round(lr_cv_mean, 4), "cv_auc_std": round(lr_cv_std, 4)},
        "Decision Tree":       {"cv_auc_mean": round(dt_cv_mean, 4), "cv_auc_std": round(dt_cv_std, 4)},
        "XGBoost":             {"cv_auc_mean": round(xgb_cv_mean, 4), "cv_auc_std": round(xgb_cv_std, 4)},
    }

    # --- Calibrate the black-box winner for honest probabilities ---
    logger.info("Calibrating XGBoost probabilities (isotonic)...")
    xgb_calibrated = calibrate(xgb_model, X_train, y_train)

    # --- Cost-optimal threshold (on calibrated probs, from train CV) ---
    y_train_prob = xgb_calibrated.predict_proba(X_train)[:, 1]
    tuned_threshold = find_cost_optimal_threshold(y_train, y_train_prob)

    # --- Evaluate on the held-out test split ---
    results: list[dict[str, Any]] = []
    results.append(evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression", 0.5))
    results.append(evaluate_model(dt_model, X_test, y_test, "Decision Tree", 0.5))
    results.append(evaluate_model(xgb_model, X_test, y_test, "XGBoost (raw)", 0.5))
    results.append(evaluate_model(xgb_calibrated, X_test, y_test, "XGBoost (calibrated)", tuned_threshold))

    for r in results:
        r.update(cv_results.get(r["model"].replace(" (raw)", "").replace(" (calibrated)", ""), {}))

    results_df = pd.DataFrame(results).set_index("model")
    logger.info("\n%s\n%s", "MODEL COMPARISON", results_df.to_string())
    best = results_df["roc_auc"].idxmax()
    logger.info("Best model by test ROC-AUC: %s (%.4f)", best, results_df.loc[best, "roc_auc"])

    # --- Persist artifacts ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(lr_model, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(dt_model, MODELS_DIR / "decision_tree.joblib")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.joblib")
    joblib.dump(xgb_calibrated, MODELS_DIR / "xgboost_calibrated.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(preproc, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(list(X_train.columns), MODELS_DIR / "feature_names.joblib")

    # Save the tuned threshold next to the model
    with open(MODELS_DIR / "threshold.json", "w") as f:
        json.dump(
            {"tuned_threshold": tuned_threshold,
             "default_threshold": DEFAULT_THRESHOLD,
             "cost_false_approve": COST_FALSE_APPROVE,
             "cost_false_deny": COST_FALSE_DENY},
            f, indent=2,
        )

    results_df.to_csv(MODEL_OUTPUT_DIR / "model_comparison.csv")
    with open(MODEL_OUTPUT_DIR / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    _append_run_log(MODEL_OUTPUT_DIR, {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "xgb_params": json.dumps(xgb_params),
        "xgb_cv_auc_mean": round(xgb_cv_mean, 4),
        "xgb_cv_auc_std": round(xgb_cv_std, 4),
        "xgb_test_auc": results_df.loc["XGBoost (calibrated)", "roc_auc"],
        "threshold": tuned_threshold,
        "n_train": len(X_train),
        "n_test": len(X_test),
    })

    logger.info("Models saved to %s", MODELS_DIR)
    logger.info("Results saved to %s", MODEL_OUTPUT_DIR)

    return TrainingArtifacts(
        xgb_model=xgb_model,
        xgb_calibrated=xgb_calibrated,
        lr_model=lr_model,
        dt_model=dt_model,
        preprocessor=preproc,
        scaler=scaler,
        feature_names=list(X_train.columns),
        results=results_df,
        tuned_threshold=tuned_threshold,
        cv_results=cv_results,
    )


if __name__ == "__main__":
    configure_logging()
    train_and_compare()
