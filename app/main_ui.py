"""Streamlit dashboard for the Glass Box Loan Classifier.

A loan officer can:
1. Input applicant data (with logical consistency checks)
2. Pick which of the three candidate models to run
3. Slide the decision threshold and see the verdict update
4. See ``Pass``/``Fail`` with a *calibrated* default-risk percentage
5. Toggle "Why?" — SHAP or LIME explanations, plus adverse-action reasons
6. Toggle "How to improve?" — DiCE counterfactuals with immutability constraints
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    COST_FALSE_APPROVE,
    COST_FALSE_DENY,
    DEFAULT_THRESHOLD,
    EDA_OUTPUT_DIR,
    EXPLAIN_OUTPUT_DIR,
    MODEL_OUTPUT_DIR,
    MODELS_DIR,
)
from src.data_ingestion import FEATURE_DESCRIPTIONS, load_and_split  # noqa: E402
from src.preprocessing import Preprocessor, get_feature_matrix  # noqa: E402
from src.reason_codes import generate_reason_codes  # noqa: E402
from src.xai_engine import (  # noqa: E402
    compute_shap_values,
    create_dice_explainer,
    create_lime_explainer,
    explain_with_lime,
    format_counterfactual_advice,
    generate_counterfactuals,
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    calibrated_path = MODELS_DIR / "xgboost_calibrated.joblib"
    xgb_model = joblib.load(MODELS_DIR / "xgboost.joblib")
    xgb_calibrated = joblib.load(calibrated_path) if calibrated_path.exists() else None
    lr_model = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    dt_model = joblib.load(MODELS_DIR / "decision_tree.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")

    preproc_path = MODELS_DIR / "preprocessor.joblib"
    preproc = joblib.load(preproc_path) if preproc_path.exists() else None

    return xgb_model, xgb_calibrated, lr_model, dt_model, scaler, feature_names, preproc


@st.cache_resource
def load_training_data(_preproc: Preprocessor | None):
    train_df, _ = load_and_split()
    if _preproc is not None:
        train_clean = _preproc.transform(train_df)
    else:
        # Fallback for older artifacts — fit fresh (may differ slightly from
        # the preprocessor used at training time).
        train_clean = Preprocessor().fit_transform(train_df)
    return get_feature_matrix(train_clean)


@st.cache_resource
def get_lime_exp(_X_train):
    return create_lime_explainer(_X_train)


@st.cache_resource
def get_dice_exp(_model, _X_train):
    return create_dice_explainer(_model, _X_train, method="genetic")


@st.cache_resource
def load_model_comparison():
    path = MODEL_OUTPUT_DIR / "model_comparison.csv"
    return pd.read_csv(path, index_col=0) if path.exists() else None


@st.cache_resource
def load_consistency_stats():
    path = EXPLAIN_OUTPUT_DIR / "consistency.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_resource
def load_threshold_config():
    path = MODELS_DIR / "threshold.json"
    if path.exists():
        return json.loads(path.read_text())
    return {
        "tuned_threshold": DEFAULT_THRESHOLD,
        "default_threshold": DEFAULT_THRESHOLD,
        "cost_false_approve": COST_FALSE_APPROVE,
        "cost_false_deny": COST_FALSE_DENY,
    }


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_prediction_result(prediction: int, probability: float, threshold: float) -> None:
    badge = f"Default risk: {probability:.1%} — threshold: {threshold:.0%}"
    if prediction == 0:
        st.success(f"**APPROVED** — {badge}", icon="\u2705")
    else:
        st.error(f"**DENIED** — {badge}", icon="\u274c")


def validate_inputs(past_due_30: int, past_due_60: int, past_due_90: int) -> list[str]:
    """Logical consistency: 60+ buckets should be subsets of 30+, etc.

    The Kaggle dataset sometimes violates this, so we warn rather than block,
    but flagging it to the loan officer is the honest thing to do.
    """
    warnings: list[str] = []
    if past_due_60 > past_due_30:
        warnings.append(
            "60-89-day past-due count is greater than the 30-59-day count — "
            "usually the 30+ bucket should include the 60+ occurrences."
        )
    if past_due_90 > past_due_60:
        warnings.append(
            "90+-day late count exceeds the 60-89-day count — "
            "check whether these were entered correctly."
        )
    return warnings


def render_reason_codes(
    shap_instance: shap.Explanation,
    instance: pd.Series,
    prediction: int,
) -> None:
    """Render plain-English reasons for *this specific applicant's* decision.

    For denials this mirrors the adverse-action format required by ECOA/Reg B;
    for approvals it lists the top positive signals so the officer has a
    matching narrative. Always render this BEFORE any SHAP/LIME visualisation —
    humans read the 'why' first, then inspect the chart for confirmation.
    """
    if prediction == 1:  # Denied
        reasons = generate_reason_codes(shap_instance, instance, top_k=4,
                                        direction="adverse")
        if not reasons:
            st.info(
                "No single adverse factor dominated — the denial is driven by "
                "a diffuse combination of smaller signals."
            )
            return
        st.markdown("#### Why this application was **denied**")
        st.markdown(
            "The principal factors weighing against approval include:"
        )
    else:  # Approved
        reasons = generate_reason_codes(shap_instance, instance, top_k=4,
                                        direction="favourable")
        if not reasons:
            st.info("No strong positive signals identified — the approval is driven by many small favourable factors.")
            return
        st.markdown("#### Why this application was **approved**")
        st.markdown("The principal factors supporting approval:")

    for r in reasons:
        st.markdown(f"- {r.description}")


def render_shap_chart_and_table(
    shap_values: shap.Explanation, instance_df: pd.DataFrame,
) -> None:
    """Render the SHAP waterfall chart and the feature-contribution table.

    Split from the reason-code narrative so the UI can present the text
    explanation first and the visualisation second.
    """
    st.markdown("**Feature-level breakdown**")
    st.caption(
        "The waterfall below traces how each feature moved this applicant's "
        "predicted default probability away from the dataset baseline. Bars "
        "to the right push toward denial; bars to the left push toward approval. "
        "The table beneath sorts the same values by magnitude."
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("SHAP Feature Contributions")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close("all")

    sv = shap_values[0]
    # Both sv.values and sv.feature_names are numpy arrays already; instance_df
    # is a DataFrame so we use .to_numpy() for the per-feature values.
    importance = pd.DataFrame({
        "Feature": sv.feature_names,
        "SHAP Value": sv.values,  # noqa: PD011 — attribute on shap Explanation, not DataFrame
        "Feature Value": instance_df.iloc[0].to_numpy(),
    }).sort_values("SHAP Value", key=abs, ascending=False)
    st.dataframe(importance, width="stretch", hide_index=True)


def render_lime_explanation(lime_explainer, model, instance_values: np.ndarray) -> None:
    exp = explain_with_lime(lime_explainer, model, instance_values)
    fig = exp.as_pyplot_figure()
    fig.suptitle("LIME Feature Contributions", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_counterfactuals(dice_explainer, instance_df: pd.DataFrame,
                           feature_names: list[str]) -> None:
    with st.spinner("Generating counterfactual scenarios..."):
        try:
            cfs = generate_counterfactuals(dice_explainer, instance_df, total_cfs=4)
        except Exception as e:
            st.warning(f"Could not generate counterfactuals: {e}")
            return

    cf_df = cfs.cf_examples_list[0].final_cfs_df
    if cf_df is None or len(cf_df) == 0:
        st.warning("Could not generate counterfactuals for this applicant.")
        return

    st.markdown("#### Actionable paths to approval:")
    original = instance_df.iloc[0]
    for i, (_, cf_row) in enumerate(cf_df.iterrows()):
        advice = format_counterfactual_advice(original, cf_row, feature_names)
        with st.expander(f"Scenario {i + 1}", expanded=(i == 0)):
            if advice:
                for a in advice:
                    st.markdown(f"- {a}")
            else:
                st.info("This scenario is borderline — no actionable changes identified.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Glass Box Loan Classifier",
        page_icon="\U0001f3e6",
        layout="wide",
    )
    st.title("\U0001f3e6 Glass Box Loan Classifier")
    st.caption("Explainable AI for credit risk assessment")

    xgb_model, xgb_calibrated, lr_model, dt_model, scaler, feature_names, preproc = load_models()
    X_train, y_train = load_training_data(preproc)
    lime_explainer = get_lime_exp(X_train)

    threshold_cfg = load_threshold_config()

    # ---- Sidebar: model selector, threshold, comparison table ----
    with st.sidebar:
        st.header("Prediction settings")
        model_options = {
            "XGBoost (calibrated)": ("xgboost_calibrated", xgb_calibrated),
            "XGBoost (raw)": ("xgboost", xgb_model),
            "Logistic Regression (glass box)": ("logistic_regression", lr_model),
            "Decision Tree (glass box)": ("decision_tree", dt_model),
        }
        # Drop the calibrated entry if artifact missing
        if xgb_calibrated is None:
            model_options.pop("XGBoost (calibrated)")

        next(iter(model_options))
        chosen_label = st.selectbox(
            "Model",
            options=list(model_options.keys()),
            index=0,
            help="XGBoost (calibrated) produces meaningful default-probability percentages."
        )
        chosen_key, chosen_model = model_options[chosen_label]
        st.caption(
            "Glass-box models are directly inspectable; black-box models "
            "rely on SHAP/LIME for explanation."
        )

        tuned = float(threshold_cfg.get("tuned_threshold", DEFAULT_THRESHOLD))
        threshold = st.slider(
            "Decision threshold",
            min_value=0.05, max_value=0.95, value=tuned, step=0.01,
            help=(
                f"Probability above this cutoff is classed as 'Denied'. "
                f"Cost-optimal value (FA={COST_FALSE_APPROVE}, FD={COST_FALSE_DENY}) "
                f"is {tuned:.2f}."
            ),
        )
        if abs(threshold - tuned) < 0.005:
            st.caption(f"Using cost-optimal threshold ({tuned:.2f}).")
        elif threshold < tuned:
            st.caption("Below cost-optimal — approving more risk.")
        else:
            st.caption("Above cost-optimal — rejecting more applicants.")

        st.divider()
        st.header("Model comparison")
        comparison = load_model_comparison()
        if comparison is not None:
            st.dataframe(comparison, width="stretch")
        else:
            st.info("Run training to populate this table.")

        st.divider()
        st.header("About")
        st.markdown(
            "**Explainable AI (XAI)** for loan approval.\n\n"
            "- **SHAP** for global & local explanations\n"
            "- **LIME** for alternative local explanations\n"
            "- **DiCE** for counterfactual recourse\n"
            "- **Isotonic calibration** so 'Default risk: X%' is honest\n"
            "- **Cost-aware threshold** tuned from training data\n\n"
            "Age and past delinquency are held fixed in recourse suggestions."
        )

    # ---- Main tabs ----
    tab_single, tab_insights, tab_data, tab_drift = st.tabs([
        "\U0001f464 Single Applicant",
        "\U0001f4ca Model Insights",
        "\U0001f50d Data Explorer",
        "\U0001f4c8 Drift Monitor",
    ])

    # -------- Tab: single applicant --------
    with tab_single:
        st.subheader("Applicant Information")
        st.markdown("Enter the applicant's financial details, or load an example.")

        col_ex1, col_ex2, col_ex3 = st.columns(3)
        with col_ex1:
            load_approved = st.button("Load example: Likely Approved")
        with col_ex2:
            load_denied = st.button("Load example: Likely Denied")
        with col_ex3:
            load_borderline = st.button("Load example: Borderline")

        if load_approved:
            defaults = {"revolving_util": 0.15, "age": 45, "past_due_30": 0,
                        "debt_ratio": 0.3, "monthly_income": 8500.0,
                        "open_credit_lines": 8, "past_due_90": 0,
                        "real_estate_loans": 2, "past_due_60": 0, "dependents": 1}
        elif load_denied:
            # Severe-risk profile: over-utilised credit, very low income, heavy
            # delinquency history. Chosen to score denied (p >= 0.5) across
            # XGBoost, calibrated XGBoost, and the glass-box trees, so the
            # verdict stays "Denied" regardless of which model or threshold
            # the user picks.
            defaults = {"revolving_util": 1.3, "age": 27, "past_due_30": 6,
                        "debt_ratio": 2.0, "monthly_income": 1600.0,
                        "open_credit_lines": 2, "past_due_90": 3,
                        "real_estate_loans": 0, "past_due_60": 4, "dependents": 4}
        elif load_borderline:
            defaults = {"revolving_util": 0.6, "age": 38, "past_due_30": 1,
                        "debt_ratio": 0.8, "monthly_income": 4500.0,
                        "open_credit_lines": 5, "past_due_90": 0,
                        "real_estate_loans": 1, "past_due_60": 0, "dependents": 2}
        else:
            defaults = {"revolving_util": 0.5, "age": 40, "past_due_30": 0,
                        "debt_ratio": 0.5, "monthly_income": 5000.0,
                        "open_credit_lines": 6, "past_due_90": 0,
                        "real_estate_loans": 1, "past_due_60": 0, "dependents": 1}

        col1, col2 = st.columns(2)
        with col1:
            revolving_util = st.number_input(
                "Credit Utilization Ratio", min_value=0.0, max_value=1.5,
                value=defaults["revolving_util"], step=0.01,
                help=FEATURE_DESCRIPTIONS["RevolvingUtilizationOfUnsecuredLines"],
            )
            age = st.number_input("Age", min_value=18, max_value=109, value=defaults["age"],
                                  help=FEATURE_DESCRIPTIONS["age"])
            past_due_30 = st.number_input(
                "Times 30-59 Days Past Due", min_value=0, max_value=20, value=defaults["past_due_30"],
                help=FEATURE_DESCRIPTIONS["NumberOfTime30-59DaysPastDueNotWorse"],
            )
            debt_ratio = st.number_input(
                "Debt Ratio", min_value=0.0, max_value=5.0, value=defaults["debt_ratio"], step=0.01,
                help=FEATURE_DESCRIPTIONS["DebtRatio"],
            )
            monthly_income = st.number_input(
                "Monthly Income ($)", min_value=0.0, max_value=500000.0,
                value=defaults["monthly_income"], step=100.0,
                help=FEATURE_DESCRIPTIONS["MonthlyIncome"],
            )
        with col2:
            open_credit_lines = st.number_input(
                "Open Credit Lines & Loans", min_value=0, max_value=60,
                value=defaults["open_credit_lines"],
                help=FEATURE_DESCRIPTIONS["NumberOfOpenCreditLinesAndLoans"],
            )
            past_due_90 = st.number_input(
                "Times 90+ Days Late", min_value=0, max_value=20, value=defaults["past_due_90"],
                help=FEATURE_DESCRIPTIONS["NumberOfTimes90DaysLate"],
            )
            real_estate_loans = st.number_input(
                "Real Estate Loans", min_value=0, max_value=30,
                value=defaults["real_estate_loans"],
                help=FEATURE_DESCRIPTIONS["NumberRealEstateLoansOrLines"],
            )
            past_due_60 = st.number_input(
                "Times 60-89 Days Past Due", min_value=0, max_value=20, value=defaults["past_due_60"],
                help=FEATURE_DESCRIPTIONS["NumberOfTime60-89DaysPastDueNotWorse"],
            )
            dependents = st.number_input(
                "Number of Dependents", min_value=0, max_value=20, value=defaults["dependents"],
                help=FEATURE_DESCRIPTIONS["NumberOfDependents"],
            )

        # Logical consistency warnings
        for w in validate_inputs(past_due_30, past_due_60, past_due_90):
            st.warning(w, icon="\u26A0")

        raw_input = pd.DataFrame([{
            "RevolvingUtilizationOfUnsecuredLines": revolving_util,
            "age": float(age),
            "NumberOfTime30-59DaysPastDueNotWorse": float(past_due_30),
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": float(open_credit_lines),
            "NumberOfTimes90DaysLate": float(past_due_90),
            "NumberRealEstateLoansOrLines": float(real_estate_loans),
            "NumberOfTime60-89DaysPastDueNotWorse": float(past_due_60),
            "NumberOfDependents": float(dependents),
        }])[feature_names]

        # Apply the same preprocessing the models were trained on:
        # median imputation, outlier capping, past-due sentinel remapping.
        # Without this, a user entering 96/98 (dataset's "unknown" sentinels)
        # or an extreme DebtRatio would be scored on out-of-distribution data.
        input_data = preproc.transform(raw_input) if preproc is not None else raw_input

        # Linear models also need scaling on top of preprocessing.
        scale_needed = chosen_key == "logistic_regression"
        input_for_model = (
            pd.DataFrame(scaler.scaler_.transform(input_data), columns=feature_names)
            if scale_needed else input_data
        )

        st.divider()
        if st.button("Evaluate Application", type="primary", width="stretch"):
            probability = float(chosen_model.predict_proba(input_for_model)[0][1])
            prediction = int(probability >= threshold)
            render_prediction_result(prediction, probability, threshold)

            st.session_state["last_input"] = input_data
            st.session_state["last_prediction"] = prediction
            st.session_state["last_probability"] = probability
            st.session_state["last_model_key"] = chosen_key
            st.session_state["last_model_label"] = chosen_label

        if "last_prediction" in st.session_state:
            st.divider()
            col_why, col_how = st.columns(2)
            with col_why:
                if st.button("Why? (SHAP Explanation)", width="stretch"):
                    st.session_state["show_shap"] = not st.session_state.get("show_shap", False)
                if st.button("Why? (LIME Explanation)", width="stretch"):
                    st.session_state["show_lime"] = not st.session_state.get("show_lime", False)
            with col_how:
                if st.session_state["last_prediction"] == 1:
                    if st.button("How to improve? (Counterfactuals)", width="stretch"):
                        st.session_state["show_cf"] = not st.session_state.get("show_cf", False)
                else:
                    st.info("Applicant was approved — no improvement suggestions needed.")

            if st.session_state.get("show_shap"):
                st.subheader("Why this decision? (SHAP)")
                # SHAP only works on tree models, not the LR/DT wrappers in this tab
                shap_model_key = st.session_state["last_model_key"]
                if shap_model_key in ("xgboost", "xgboost_calibrated"):
                    # 1. Compute SHAP values once, ahead of anything visual.
                    shap_values = compute_shap_values(
                        xgb_calibrated if shap_model_key == "xgboost_calibrated" else xgb_model,
                        st.session_state["last_input"],
                    )

                    # 2. Lead with the plain-English reasons — the narrative
                    #    comes first, the chart and table come after as a
                    #    visual confirmation.
                    render_reason_codes(
                        shap_values[0],
                        st.session_state["last_input"].iloc[0],
                        st.session_state["last_prediction"],
                    )

                    # 3. Then the waterfall chart + feature-contribution table.
                    st.divider()
                    render_shap_chart_and_table(
                        shap_values, st.session_state["last_input"],
                    )
                else:
                    st.info(
                        "SHAP waterfall is shown for tree models. "
                        "For Logistic Regression, use the 'Why? (LIME Explanation)' "
                        "button or consult the raw model coefficients."
                    )

            if st.session_state.get("show_lime"):
                st.subheader("Why this decision? (LIME)")
                verdict = "denied" if st.session_state["last_prediction"] == 1 else "approved"
                st.markdown(
                    f"LIME fits a simple, interpretable model in a small "
                    f"neighbourhood around this applicant and uses its "
                    f"coefficients to explain why the decision came out as "
                    f"**{verdict}**. Bars on the left favour approval; bars "
                    f"on the right push toward denial."
                )
                render_lime_explanation(
                    lime_explainer, chosen_model,
                    st.session_state["last_input"].iloc[0].values,
                )

            if st.session_state.get("show_cf") and st.session_state["last_prediction"] == 1:
                st.subheader("How to Improve (Counterfactual Explanations)")
                st.markdown(
                    "These scenarios show what changes would flip the decision to **approval**. "
                    "Age and past delinquency history are held fixed."
                )
                # DiCE is wired to the tree model for speed
                dice_explainer = get_dice_exp(xgb_model, X_train)
                render_counterfactuals(
                    dice_explainer, st.session_state["last_input"], feature_names,
                )

    # -------- Tab: model insights --------
    with tab_insights:
        st.subheader("Global Feature Importance (SHAP)")
        shap_summary_path = EXPLAIN_OUTPUT_DIR / "shap_summary.png"
        shap_bar_path = EXPLAIN_OUTPUT_DIR / "shap_bar.png"
        pdp_path = EXPLAIN_OUTPUT_DIR / "partial_dependence.png"

        if shap_summary_path.exists() and shap_bar_path.exists():
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.image(str(shap_summary_path), caption="SHAP Beeswarm Plot")
            with col_s2:
                st.image(str(shap_bar_path), caption="SHAP Mean Absolute Importance")
        else:
            st.warning("Run `python -m src.xai_engine` to generate SHAP plots.")

        if pdp_path.exists():
            st.divider()
            st.subheader("Partial Dependence Plots")
            st.markdown("Average shape of each feature's effect on P(default).")
            st.image(str(pdp_path))

        st.divider()
        st.subheader("Model Performance Comparison")
        comparison = load_model_comparison()
        if comparison is not None:
            st.bar_chart(comparison.select_dtypes(include="number"))
        else:
            st.warning("Run training to populate this chart.")

        st.divider()
        st.subheader("SHAP vs LIME Consistency")
        stats = load_consistency_stats()
        if stats is not None:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric(
                f"Mean agreement (top-{stats['top_k']})",
                f"{stats['mean_agreement']:.1%}",
            )
            col_b.metric("Std", f"{stats['std_agreement']:.1%}")
            col_c.metric("n samples", stats["n_samples"])
            st.caption(f"Verdict: {stats.get('verdict', 'n/a')}")
        else:
            st.info("Run `python -m src.xai_engine` to compute the agreement metric.")

    # -------- Tab: data explorer --------
    with tab_data:
        st.subheader("Exploratory Data Analysis")
        corr_path = EDA_OUTPUT_DIR / "correlation_matrix.png"
        dist_path = EDA_OUTPUT_DIR / "feature_distributions.png"
        target_corr_path = EDA_OUTPUT_DIR / "target_correlation.png"

        if corr_path.exists():
            st.image(str(corr_path), caption="Feature Correlation Matrix (Proxy Variable Check)")
            st.markdown(
                "**Key finding:** the three past-due features are highly correlated. "
                "Since they all measure delinquency behaviour, this correlation is "
                "expected; in a deployment with protected attributes you would also "
                "check correlation of each feature with those attributes."
            )
        if target_corr_path.exists():
            st.image(str(target_corr_path), caption="Feature Correlation with Loan Default")
        if dist_path.exists():
            st.image(str(dist_path), caption="Feature Distributions by Target Class")
        if not any(p.exists() for p in [corr_path, dist_path, target_corr_path]):
            st.warning("Run `python -m src.eda` to generate EDA plots.")

    # -------- Tab: drift monitor --------
    with tab_drift:
        st.subheader("Population Stability Index (PSI)")
        st.markdown(
            "PSI flags whether a feature's distribution has shifted between "
            "the training population and a new population. Common thresholds: "
            "< 0.10 OK, 0.10–0.25 moderate, ≥ 0.25 major. Baseline below "
            "compares train vs test (should be near zero)."
        )
        from src.config import DRIFT_OUTPUT_DIR  # local import to avoid cycles
        psi_path = DRIFT_OUTPUT_DIR / "psi_report.csv"
        if psi_path.exists():
            psi = pd.read_csv(psi_path)
            st.dataframe(psi, width="stretch", hide_index=True)
        else:
            st.info("Run `python -m src.drift` to compute the PSI report.")


if __name__ == "__main__":
    main()
