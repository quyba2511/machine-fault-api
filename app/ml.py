from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from app.schemas import PredictionRequest

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fault_model.joblib"
MODEL_VERSION = "v1.0"
THRESHOLD = 0.5

FEATURES = [
    "temperature",
    "pressure",
    "vibration",
    "run_hours",
    "alarm_count",
    "maintenance_days",
]


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run `python -m training.train` first."
        )
    return joblib.load(MODEL_PATH)


def get_risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "HIGH"
    if probability >= 0.5:
        return "MEDIUM"
    return "LOW"


def predict_fault(data: PredictionRequest) -> dict:
    input_data = {
        "temperature": data.temperature,
        "pressure": data.pressure,
        "vibration": data.vibration,
        "run_hours": data.run_hours,
        "alarm_count": data.alarm_count,
        "maintenance_days": data.maintenance_days,
    }
    frame = pd.DataFrame([input_data], columns=FEATURES)
    probability = float(load_model().predict_proba(frame)[0, 1])

    return {
        "machine_id": data.machine_id,
        "fault_probability": round(probability, 4),
        "prediction": int(probability >= THRESHOLD),
        "risk_level": get_risk_level(probability),
        "model_version": MODEL_VERSION,
    }
