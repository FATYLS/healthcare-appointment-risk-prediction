from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = {
    "PatientId", "AppointmentID", "Gender", "ScheduledDay", "AppointmentDay",
    "Age", "Neighbourhood", "Scholarship", "Hipertension", "Diabetes",
    "Alcoholism", "Handcap", "SMS_received", "No-show"
}


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    out["ScheduledDay"] = pd.to_datetime(out["ScheduledDay"], errors="coerce", utc=True)
    out["AppointmentDay"] = pd.to_datetime(out["AppointmentDay"], errors="coerce", utc=True)
    out = out.dropna(subset=["ScheduledDay", "AppointmentDay", "Age"])
    out = out[out["Age"].between(0, 120)].copy()
    out["target_no_show"] = (out["No-show"].astype(str).str.lower() == "yes").astype(int)
    return out
