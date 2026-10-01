# Model artifacts

The trained `.joblib` artifact is intentionally not committed by default. Generate it locally with:

```bash
python -m src.train
```

This creates `models/best_model.joblib` and updates `models/metrics.json`.
