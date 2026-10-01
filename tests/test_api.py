from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def payload():
    return {
        "Age": 42,
        "gender": "F",
        "waiting_days": 7,
        "scheduled_hour": 10,
        "scheduled_day_of_week": 1,
        "appointment_day_of_week": 2,
        "appointment_month": 5,
        "is_weekend": 0,
        "Scholarship": 0,
        "Hipertension": 0,
        "Diabetes": 0,
        "Alcoholism": 0,
        "Handcap": 0,
        "SMS_received": 1,
    }


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_predict_or_missing_model():
    response = client.post("/predict", json=payload())
    assert response.status_code in {200, 503}
