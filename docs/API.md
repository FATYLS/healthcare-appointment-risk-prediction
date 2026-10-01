# API

Run locally after generating the model artifact:

```bash
python -m src.train
uvicorn api.main:app --reload
```

Open `/docs` for the generated OpenAPI documentation.

Endpoints:

- `GET /health`
- `POST /predict`

The prediction endpoint returns a no-show probability, a simple risk level, the threshold used and the model version. It does not return patient identifiers.
