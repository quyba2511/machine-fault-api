from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import joblib
import pandas as pd

from app.schemas import PredictionRequest

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "fault_model.joblib"
DEFAULT_MODEL_VERSION = "v1.0"
DEFAULT_THRESHOLD = 0.5

FEATURES = [
    "temperature",
    "pressure",
    "vibration",
    "run_hours",
    "alarm_count",
    "maintenance_days",
]


@dataclass(frozen=True)
class LoadedModel:
    model: object
    version: str
    threshold: float
    artifact_path: str


class ModelRuntime:
    def __init__(self) -> None:
        self._lock = RLock()
        self._loaded: LoadedModel | None = None

    def current(self) -> LoadedModel:
        with self._lock:
            if self._loaded is None:
                self._loaded = self._load_artifact(
                    DEFAULT_MODEL_PATH,
                    DEFAULT_MODEL_VERSION,
                    DEFAULT_THRESHOLD,
                )
            return self._loaded

    def reload(self, artifact_path: str, version: str, threshold: float) -> LoadedModel:
        candidate = self._load_artifact(Path(artifact_path), version, threshold)
        # Load and validate first. Only then swap the shared reference.
        with self._lock:
            self._loaded = candidate
            return self._loaded

    @staticmethod
    def _load_artifact(path: Path, version: str, threshold: float) -> LoadedModel:
        if not path.exists():
            raise RuntimeError(f"Model artifact not found: {path}")
        model = joblib.load(path)
        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Model artifact does not support predict_proba")
        return LoadedModel(model=model, version=version, threshold=float(threshold), artifact_path=str(path))


runtime = ModelRuntime()


def load_model():
    return runtime.current().model


def get_runtime_info() -> dict:
    loaded = runtime.current()
    return {
        "model_version": loaded.version,
        "threshold": loaded.threshold,
        "artifact_path": loaded.artifact_path,
    }


def get_risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "HIGH"
    if probability >= 0.5:
        return "MEDIUM"
    return "LOW"


def predict_fault(data: PredictionRequest) -> dict:
    loaded = runtime.current()
    frame = pd.DataFrame([{
        "temperature": data.temperature,
        "pressure": data.pressure,
        "vibration": data.vibration,
        "run_hours": data.run_hours,
        "alarm_count": data.alarm_count,
        "maintenance_days": data.maintenance_days,
    }], columns=FEATURES)
    probability = float(loaded.model.predict_proba(frame)[0, 1])
    return {
        "machine_id": data.machine_id,
        "fault_probability": round(probability, 4),
        "prediction": int(probability >= loaded.threshold),
        "risk_level": get_risk_level(probability),
        "model_version": loaded.version,
    }
