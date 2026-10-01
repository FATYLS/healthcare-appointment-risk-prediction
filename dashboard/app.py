import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/best_model.joblib"
METRICS_PATH = ROOT / "models/metrics.json"

st.set_page_config(page_title="Appointment Risk Prediction", page_icon="🏥", layout="wide")
st.title("Healthcare Appointment Risk Prediction")
st.caption("Data Science portfolio prototype — not a medical decision system.")

if not METRICS_PATH.exists():
    st.warning("Training artifacts are missing. Run `python -m src.train` first.")
    st.stop()

metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
tabs = st.tabs(["Overview", "Risk prediction", "Model performance", "Responsible AI"])

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Appointments", f"{metrics['dataset_rows_after_cleaning']:,}")
    c2.metric("No-show rate", f"{metrics['no_show_rate']:.1%}")
    c3.metric("Best model", metrics["best_model"])
    st.write("The project uses a chronological split to reduce temporal leakage and compares multiple classifiers.")

with tabs[1]:
    age = st.slider("Age", 0, 100, 40)
    gender = st.selectbox("Gender", ["F", "M"])
    waiting_days = st.slider("Waiting days", 0, 180, 7)
    scheduled_hour = st.slider("Scheduled hour", 0, 23, 10)
    scheduled_day = st.selectbox("Scheduled weekday", range(7), 1)
    appointment_day = st.selectbox("Appointment weekday", range(7), 1)
    month = st.slider("Appointment month", 1, 12, 5)
    sms = st.selectbox("SMS received", [0, 1])
    scholarship = st.selectbox("Scholarship", [0, 1])
    hypertension = st.selectbox("Hypertension", [0, 1])
    diabetes = st.selectbox("Diabetes", [0, 1])
    alcoholism = st.selectbox("Alcoholism", [0, 1])
    handicap = st.selectbox("Handicap", [0, 1])

    if st.button("Predict risk"):
        if not MODEL_PATH.exists():
            st.error("Model artifact not available. Run the training pipeline first.")
        else:
            model = joblib.load(MODEL_PATH)
            row = pd.DataFrame([{
                "Age": age,
                "gender_male": int(gender == "M"),
                "waiting_days": waiting_days,
                "scheduled_hour": scheduled_hour,
                "scheduled_day_of_week": scheduled_day,
                "appointment_day_of_week": appointment_day,
                "appointment_month": month,
                "is_weekend": int(appointment_day >= 5),
                "Scholarship": scholarship,
                "Hipertension": hypertension,
                "Diabetes": diabetes,
                "Alcoholism": alcoholism,
                "Handcap": handicap,
                "SMS_received": sms,
            }])
            probability = float(model.predict_proba(row)[:, 1][0])
            st.metric("Estimated no-show probability", f"{probability:.1%}")
            st.write("Risk:", "**HIGH**" if probability >= 0.50 else "**LOW**")

with tabs[2]:
    rows = []
    for name, result in metrics["models"].items():
        test = result["test"]
        rows.append({
            "Model": name,
            "Threshold": test["threshold"],
            "Precision": test["precision"],
            "Recall": test["recall"],
            "F1": test["f1"],
            "ROC-AUC": test["roc_auc"],
            "PR-AUC": test["pr_auc"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tabs[3]:
    st.markdown(
        """
        - Patient and appointment identifiers are excluded from model features.
        - Predictions are operational risk estimates, not diagnoses.
        - Historical data may not generalize to another healthcare system.
        - A high risk score must never be used to deny or clinically deprioritize care.
        - Real deployment would require privacy, security, governance and subgroup-performance review.
        """
    )
