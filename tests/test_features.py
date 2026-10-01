import pandas as pd

from src.features import MODEL_FEATURES, build_features


def test_features():
    data = pd.DataFrame({
        "ScheduledDay": pd.to_datetime(["2016-05-01T10:00:00Z"]),
        "AppointmentDay": pd.to_datetime(["2016-05-03T00:00:00Z"]),
        "Age": [40],
        "Gender": ["F"],
        "Scholarship": [0],
        "Hipertension": [1],
        "Diabetes": [0],
        "Alcoholism": [0],
        "Handcap": [0],
        "SMS_received": [1],
        "target_no_show": [0],
    })
    result = build_features(data)
    assert list(result.columns) == MODEL_FEATURES + ["target_no_show"]
    assert abs(result.loc[0, "waiting_days"] - 1.5833333333) < 1e-6
    assert result.loc[0, "gender_male"] == 0
