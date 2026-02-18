# Glass Box Loan Classifier

Explainable AI (XAI) dashboard for credit scoring models.

## Project Overview

Finance models are often "black boxes," making it hard to explain to a customer why their loan was denied. This tool aims to provide transparency using SHAP and counterfactual explanations through an interactive dashboard.

## Key Features

- **Explainability**: Understand "why" a loan was approved or denied.
- **Global Explanations**: Visualize what features the model values most.
- **Local Explanations**: See why a specific person was denied.
- **Counterfactuals**: Discover exactly what changes are needed to flip the model's decision.

## Stack

- Python
- Scikit-learn
- SHAP
- Streamlit
- DiCE (for Counterfactuals)

## Directory Structure
- `data/`: Raw and processed datasets.
- `models/`: Trained model files.
- `notebooks/`: EDA and prototyping.
- `src/`: Core logic for ingestion, preprocessing, training, and XAI.
- `app/`: Streamlit UI.
- `tests/`: Unit tests.
