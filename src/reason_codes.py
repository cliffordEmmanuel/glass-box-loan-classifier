"""Adverse-action reason code generation.

US lending regulations (ECOA / Reg B, FCRA) require lenders to send an
adverse-action notice listing the principal reasons a credit application was
denied. The reasons must be specific enough to help the applicant improve,
not just "credit score too low".

This module converts the top-*k* most damaging SHAP contributions for a
denied applicant into plain-English reason strings. It is intentionally
conservative: only features that *pushed the probability toward denial* are
listed, and the phrasing is geared toward what the applicant can act on.

A real deployment would map these to the standard FCRA reason-code list;
what we produce here is the human-readable explanation that accompanies
those codes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# How each feature should be phrased to an applicant, and the direction of
# change that would *hurt* the application (the "sign of concern").
_REASON_TEMPLATES = {
    "RevolvingUtilizationOfUnsecuredLines": (
        "High credit utilisation ({value:.0%} of limit)",
        "increase",
    ),
    "DebtRatio": (
        "Debt-to-income ratio is elevated ({value:.0%})",
        "increase",
    ),
    "MonthlyIncome": (
        "Monthly income (${value:,.0f}) below the approval range for similar applications",
        "decrease",
    ),
    "NumberOfTime30-59DaysPastDueNotWorse": (
        "History of payments 30-59 days past due ({value:.0f} times)",
        "increase",
    ),
    "NumberOfTime60-89DaysPastDueNotWorse": (
        "History of payments 60-89 days past due ({value:.0f} times)",
        "increase",
    ),
    "NumberOfTimes90DaysLate": (
        "History of serious delinquency — 90+ days late ({value:.0f} times)",
        "increase",
    ),
    "NumberOfOpenCreditLinesAndLoans": (
        "Number of open credit lines ({value:.0f})",
        "either",
    ),
    "NumberRealEstateLoansOrLines": (
        "Number of real-estate loans ({value:.0f})",
        "either",
    ),
    "NumberOfDependents": (
        "Number of dependents ({value:.0f})",
        "increase",
    ),
    "age": (
        "Applicant age ({value:.0f})",
        "either",
    ),
}

# Mirror of the adverse templates, phrased positively for approval reasons.
# Used when ``direction='favourable'`` to describe features whose SHAP values
# *reduced* default probability for this applicant.
_FAVOURABLE_TEMPLATES = {
    "RevolvingUtilizationOfUnsecuredLines":
        "Conservative credit utilisation ({value:.0%} of limit)",
    "DebtRatio":
        "Healthy debt-to-income ratio ({value:.0%})",
    "MonthlyIncome":
        "Strong monthly income (${value:,.0f})",
    "NumberOfTime30-59DaysPastDueNotWorse":
        "Clean recent payment history (30-59 day late count: {value:.0f})",
    "NumberOfTime60-89DaysPastDueNotWorse":
        "No serious recent delinquency (60-89 day late count: {value:.0f})",
    "NumberOfTimes90DaysLate":
        "No history of 90+ day delinquency ({value:.0f})",
    "NumberOfOpenCreditLinesAndLoans":
        "Established credit footprint ({value:.0f} open lines)",
    "NumberRealEstateLoansOrLines":
        "Real-estate loan history ({value:.0f})",
    "NumberOfDependents":
        "Household size ({value:.0f} dependents)",
    "age":
        "Applicant age ({value:.0f})",
}


@dataclass
class ReasonCode:
    feature: str
    impact: float          # SHAP value toward denial (positive = worse)
    value: float           # the applicant's feature value
    description: str       # human-readable sentence

    def as_text(self) -> str:
        return f"- {self.description} (impact: +{self.impact:.3f})"


def generate_reason_codes(
    shap_values,
    instance: pd.Series,
    top_k: int = 3,
    direction: str = "adverse",
) -> list[ReasonCode]:
    """Return the top-k features that most influenced a single prediction.

    Parameters
    ----------
    shap_values
        A ``shap.Explanation`` object for a *single* instance (i.e.
        ``shap_values[0]``), where positive values indicate contribution
        toward the positive class ("Default").
    instance
        The raw feature values for the applicant — used to fill in the
        template placeholders.
    top_k
        How many reasons to return. Regulation typically requires up to four.
    direction
        ``"adverse"`` (default) returns features that pushed the prediction
        toward *denial* — the regulatory adverse-action reasons. ``"favourable"``
        returns features that pushed the prediction toward *approval*, phrased
        positively; use this to explain why an application was approved.
    """
    if direction not in {"adverse", "favourable"}:
        raise ValueError(f"direction must be 'adverse' or 'favourable', got {direction!r}")

    feat_names = list(shap_values.feature_names)
    sv = np.asarray(shap_values.values).ravel()

    if direction == "adverse":
        # Features that pushed toward denial (positive SHAP for class 1),
        # ranked most-harmful first.
        idx = np.where(sv > 0)[0]
        ordered = idx[np.argsort(-sv[idx])][:top_k]
    else:  # favourable
        # Features that pushed toward approval (negative SHAP for class 1),
        # ranked most-helpful first.
        idx = np.where(sv < 0)[0]
        ordered = idx[np.argsort(sv[idx])][:top_k]

    reasons: list[ReasonCode] = []
    for i in ordered:
        feat = feat_names[i]
        value = float(instance[feat]) if feat in instance else float("nan")

        if direction == "adverse":
            template, _ = _REASON_TEMPLATES.get(
                feat, (f"{feat} contributed to the decision (value: {{value}})", "either")
            )
        else:
            template = _FAVOURABLE_TEMPLATES.get(
                feat, f"{feat} supported approval (value: {{value}})"
            )

        reasons.append(
            ReasonCode(
                feature=feat,
                impact=float(sv[i]),
                value=value,
                description=template.format(value=value),
            )
        )
    return reasons
