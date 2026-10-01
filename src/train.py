from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import clean_raw, load_raw
from src.features import MODEL_FEATURES, build_features

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/raw/KaggleV2-May-2016.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def evaluate(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def best_threshold(y_true, proba):
    candidates = np.arange(0.10, 0.91, 0.01)
    return max((evaluate(y_true, proba, t) for t in candidates), key=lambda x: x["f1"])


def chronological_split(data):
    ordered = data.sort_values("AppointmentDay").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return ordered.iloc[:train_end], ordered.iloc[train_end:val_end], ordered.iloc[val_end:]


def main():
    raw = load_raw(DATA_PATH)
    clean = clean_raw(raw)
    data = build_features(clean)

    train, validation, test = chronological_split(data)
    X_train, y_train = train[MODEL_FEATURES], train["target_no_show"]
    X_val, y_val = validation[MODEL_FEATURES], validation["target_no_show"]
    X_test, y_test = test[MODEL_FEATURES], test["target_no_show"]

    models = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(random_state=42, max_iter=250)),
        ]),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        selected = best_threshold(y_val, val_proba)
        test_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "validation": selected,
            "test": evaluate(y_test, test_proba, selected["threshold"]),
        }
        fitted[name] = model

    best_name = max(results, key=lambda name: results[name]["validation"]["f1"])
    best_model = fitted[best_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    sms = clean.groupby("SMS_received")["target_no_show"].agg(["mean", "count"])
    no_sms_rate = float(sms.loc[0, "mean"])
    sms_rate = float(sms.loc[1, "mean"])

    payload = {
        "project_version": "1.0.0",
        "dataset_rows_raw": int(len(raw)),
        "dataset_columns_raw": int(raw.shape[1]),
        "dataset_rows_after_cleaning": int(len(clean)),
        "no_show_rate": float(clean["target_no_show"].mean()),
        "split": {"train": len(train), "validation": len(validation), "test": len(test), "strategy": "chronological by AppointmentDay"},
        "features": MODEL_FEATURES,
        "best_model": best_name,
        "models": results,
        "sms_analysis": {
            "no_show_rate_no_sms": no_sms_rate,
            "no_show_rate_sms": sms_rate,
            "absolute_difference": sms_rate - no_sms_rate,
        },
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
