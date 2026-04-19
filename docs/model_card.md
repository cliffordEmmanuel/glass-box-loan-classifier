# Model Card — Glass Box Loan Classifier

## Model details

- **Model name:** Glass Box Loan Classifier
- **Version:** 0.2.0
- **Developers:** Project owner (portfolio project)
- **Date:** 2026
- **Model type:** Binary classifier (probability of serious delinquency within two years)
- **Candidate estimators:** Logistic Regression, Decision Tree (depth 5), XGBoost (tuned via 5-fold CV grid search). The XGBoost model is wrapped in `CalibratedClassifierCV` (isotonic) for honest probabilities.
- **Inputs:** 10 numeric features describing an applicant's credit history, income, and household (see "Training data" below).
- **Output:** Probability of default over two years, plus a binary decision using a cost-aware threshold.
- **Licence:** Internal / educational use. The underlying dataset is covered by Kaggle's competition terms.

## Intended use

- **Primary use:** Demonstration of an explainability workflow for credit risk — SHAP (global and local), LIME, DiCE counterfactuals, adverse-action reason codes, and a SHAP/LIME consistency check.
- **Intended users:** Risk analysts and students learning XAI techniques. **Not a production lending system.**
- **Out-of-scope uses:**
  - Real-world lending decisions against any live population.
  - Decisions about individuals who differ materially from the training population (geography, time period, credit-reporting regime).
  - Any use that treats the default-probability as a causal estimate.

## Training data

- **Source:** [Kaggle — Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (2011).
- **Size:** 150,000 borrower records; ~6.7% positive class (serious delinquency within two years).
- **Features:** `RevolvingUtilizationOfUnsecuredLines`, `age`, `NumberOfTime30-59DaysPastDueNotWorse`, `DebtRatio`, `MonthlyIncome`, `NumberOfOpenCreditLinesAndLoans`, `NumberOfTimes90DaysLate`, `NumberRealEstateLoansOrLines`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfDependents`.
- **Splits:** 80/20 stratified train/test (fixed random seed). Model selection and tuning use 5-fold stratified CV on the training split only; the test split is a single held-out evaluation per model.
- **Preprocessing (learnt on train, applied to test):**
  - Median imputation for `MonthlyIncome` and `NumberOfDependents`.
  - Clip `RevolvingUtilization` to 1.5 and `DebtRatio` to 5.0 (domain-informed caps).
  - Remap past-due sentinel values 96 and 98 to the train-set median of the column.

## Evaluation

Reported metrics (see `outputs/models/model_comparison.csv` after training) are ROC-AUC, F1, and accuracy on the held-out test split plus 5-fold CV ROC-AUC mean and std on train. The dashboard reports *calibrated* default probabilities; the calibrator is fit via `CalibratedClassifierCV(method="isotonic", cv="prefit")` on the training data. Calibration on a separate holdout would be purer but costs data.

**Decision threshold.** The raw 0.5 cutoff is not business-optimal under class imbalance. The training script searches thresholds at 0.01 resolution and picks the one that minimises expected loss under the configured false-approve / false-deny cost ratio (defaults: 5.0 / 1.0). The dashboard exposes a slider so an operator can see the consequence of moving it.

## Ethical considerations and fairness

The Give Me Some Credit dataset contains **no protected attributes** (race, gender, marital status, zip code, national origin). As a result this project cannot measure demographic parity, equalised odds, or other standard group-fairness metrics. We do:

1. Run a **correlation-matrix proxy check** at EDA time, to flag features that could act as proxies for protected attributes in a real deployment.
2. Freeze **age** and **past-delinquency history** in counterfactual recourse, so the advice never suggests "be younger" or "un-do your missed payments".
3. Produce **adverse-action reason codes** for every denied application, as a starting point for ECOA / Reg B / FCRA notices.

A real deployment would additionally need:

- Linkage to protected-attribute data (with appropriate legal review) so group-level metrics can be computed.
- Disparate-impact analysis across proxy features (zip, occupation, education).
- Reject-inference handling — live data only contains approved-cohort outcomes.
- Ongoing fairness monitoring as part of the drift pipeline.

## Limitations

- **Reject inference.** The training population is only borrowers who were historically approved; the model cannot see what would have happened to rejected applicants. This biases decisions toward the historical policy.
- **Dataset age.** The data is from 2008–2011, spanning the global financial crisis. Default patterns from that window will not match current conditions.
- **Feature sparsity.** Ten features is tiny compared with modern bureau data; an operational model would have hundreds.
- **Probability calibration is not perfect.** Isotonic calibration on the training set can over-fit the calibration curve; a separate calibration split is preferred in production.
- **SHAP / LIME agreement.** The dashboard reports a measured top-5 overlap; low agreement should prompt investigation, not blind trust in either method.

## Monitoring

- Each training run appends a row to `outputs/models/run_log.csv` with timestamp, chosen hyperparameters, CV metrics, test metrics, and the tuned threshold.
- `python -m src.drift` computes Population Stability Index (PSI) per feature between any two splits. A production deployment would run it on a rolling window of scoring data against the training reference.
- Standard thresholds: `PSI < 0.10` fine, `0.10–0.25` investigate, `≥ 0.25` major shift.

## Contact

Questions or issues: file them in the project repository.
