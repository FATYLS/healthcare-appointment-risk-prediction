from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/best_model.joblib"

app = FastAPI(title="Healthcare Appointment Risk API", version="1.0.0")
MODEL = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


class Appointment(BaseModel):
    Age: int = Field(ge=0, le=120)
    gender: str
    waiting_days: float = Field(ge=0)
    scheduled_hour: int = Field(ge=0, le=23)
    scheduled_day_of_week: int = Field(ge=0, le=6)
    appointment_day_of_week: int = Field(ge=0, le=6)
    appointment_month: int = Field(ge=1, le=12)
    is_weekend: int = Field(ge=0, le=1)
    Scholarship: int = Field(ge=0, le=1)
    Hipertension: int = Field(ge=0, le=1)
    Diabetes: int = Field(ge=0, le=1)
    Alcoholism: int = Field(ge=0, le=1)
    Handcap: int = Field(ge=0, le=1)
    SMS_received: int = Field(ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "model_version": "1.0.0"}


@app.post("/predict")
def predict(p: Appointment):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model artifact not available. Run python -m src.train first.")

    payload = p.model_dump()
    row = pd.DataFrame([{
        "Age": payload["Age"],
        "gender_male": int(payload["gender"].upper() == "M"),
        "waiting_days": payload["waiting_days"],
        "scheduled_hour": payload["scheduled_hour"],
        "scheduled_day_of_week": payload["scheduled_day_of_week"],
        "appointment_day_of_week": payload["appointment_day_of_week"],
        "appointment_month": payload["appointment_month"],
        "is_weekend": payload["is_weekend"],
        "Scholarship": payload["Scholarship"],
        "Hipertension": payload["Hipertension"],
        "Diabetes": payload["Diabetes"],
        "Alcoholism": payload["Alcoholism"],
        "Handcap": payload["Handcap"],
        "SMS_received": payload["SMS_received"],
    }])

    probability = float(MODEL.predict_proba(row)[:, 1][0])
    threshold = 0.50
    return {
        "no_show_probability": round(probability, 4),
        "risk_level": "high" if probability >= threshold else "low",
        "threshold_used": threshold,
        "model_version": "1.0.0",
        "note": "Portfolio prototype; not a medical decision system.",
    }
