# Healthcare Appointment Risk Prediction

End-to-end Data Science portfolio project for predicting medical appointment no-show risk with Python and scikit-learn.

> **Portfolio / educational project. Not a medical decision system.**

## Objective
Estimate the probability that a scheduled appointment will result in a no-show and explore how predictive analytics could support appointment-management workflows.

## Dataset
This project uses the public **Medical Appointment No Shows** dataset. The raw CSV is intentionally excluded from GitHub. Place `KaggleV2-May-2016.csv` locally in `data/raw/` before training.

## Methodology
- Data quality audit and cleaning
- Temporal feature engineering
- Chronological train/validation/test split
- Logistic Regression, Random Forest and HistGradientBoosting
- Precision, Recall, F1, ROC-AUC and PR-AUC
- Threshold analysis
- Observational SMS analysis
- A/B testing design
- Model explainability
- FastAPI prediction service
- Streamlit dashboard
- Pytest and Docker support

## Test-set results
The following results were computed from the actual dataset used for the portfolio project.

| Metric | Result |
|---|---:|
| Best model | HistGradientBoosting |
| Precision | 0.279 |
| Recall | 0.760 |
| F1 | 0.408 |
| ROC-AUC | 0.727 |
| PR-AUC | 0.325 |
| Threshold | 0.50 |

The model prioritizes recall, identifying a large proportion of subsequent no-shows while accepting more false positives. The operational threshold should ultimately depend on the relative business costs of false positives and false negatives.

## SMS analysis
The historical dataset shows an association between `SMS_received` and no-show rates. This is an **observational association, not a causal result**. Reminder assignment can be confounded by appointment characteristics and operational targeting. The project therefore includes an A/B testing framework for a future randomized experiment.

## Responsible AI
- Patient and appointment identifiers are excluded from model features.
- Health-related variables require careful governance.
- Historical performance may not generalize to another population or healthcare system.
- Predictions are operational risk estimates, not diagnoses or clinical recommendations.
- Real deployment would require privacy, security, fairness, legal and human-oversight review.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
uvicorn api.main:app --reload
streamlit run dashboard/app.py
```

Tests:

```bash
pytest -q
```

## Project structure

```text
healthcare-appointment-risk-prediction/
├── api/
├── dashboard/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── models/
├── notebooks/
├── src/
└── tests/
```

## Limitations
The source data is historical and observational. It does not provide randomized treatment assignment for SMS reminders and cannot establish causal effects. This repository demonstrates a reproducible Data Science workflow rather than a validated healthcare product.
