# Responsible AI & privacy

- PatientId and AppointmentID are excluded from model features.
- Health-condition indicators are sensitive and should receive governance review before real deployment.
- The model predicts operational no-show risk, not a medical condition.
- A high-risk score must never be used to deny care or clinically deprioritize a patient.
- The dataset is historical (2016) and comes from a specific healthcare context; dataset shift is expected.
- Subgroup error rates and calibration should be reviewed before production use.
- Real deployment would require access control, retention rules, security, legal/privacy review and monitoring.
