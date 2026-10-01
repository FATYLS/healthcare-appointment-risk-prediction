# Feature engineering

Temporal features are derived from scheduling and appointment timestamps:

- waiting_days
- scheduled_hour
- scheduled_day_of_week
- appointment_day_of_week
- appointment_month
- is_weekend

Gender is binary encoded for the prototype. Patient and appointment identifiers are excluded. No post-appointment information is used as a model feature.
