# Machine Fault Prediction API

Production-oriented ML project for predicting machine fault risk.

## V1

- Logistic Regression baseline inside a scikit-learn Pipeline
- FastAPI prediction service
- Pydantic request validation
- Health endpoint
- Synthetic training data for local development
- Pytest API tests
- Docker support

## Architecture

Training data -> training pipeline -> model artifact -> FastAPI -> prediction

The model artifact is generated locally and is intentionally not committed to Git.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m training.train
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Example

POST `/predict`

```json
{
  "machine_id": "MC001",
  "temperature": 85,
  "pressure": 5.1,
  "vibration": 6.8,
  "run_hours": 5200,
  "alarm_count": 5,
  "maintenance_days": 90
}
```

## Tests

Train the model first, then run:

```bash
pytest
```

## Next milestone

V2 will persist prediction history and ground truth in SQL Server, with transaction handling and model-version tracking.
