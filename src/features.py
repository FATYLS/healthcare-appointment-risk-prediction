import pandas as pd

MODEL_FEATURES = [
    "Age", "gender_male", "waiting_days", "scheduled_hour",
    "scheduled_day_of_week", "appointment_day_of_week", "appointment_month",
    "is_weekend", "Scholarship", "Hipertension", "Diabetes", "Alcoholism",
    "Handcap", "SMS_received",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["waiting_days"] = ((x["AppointmentDay"] - x["ScheduledDay"]).dt.total_seconds() / 86400).clip(lower=0)
    x["scheduled_hour"] = x["ScheduledDay"].dt.hour
    x["scheduled_day_of_week"] = x["ScheduledDay"].dt.dayofweek
    x["appointment_day_of_week"] = x["AppointmentDay"].dt.dayofweek
    x["appointment_month"] = x["AppointmentDay"].dt.month
    x["is_weekend"] = x["appointment_day_of_week"].isin([5, 6]).astype(int)
    x["gender_male"] = (x["Gender"].astype(str).str.upper() == "M").astype(int)
    return x[MODEL_FEATURES + ["target_no_show"]].copy()
