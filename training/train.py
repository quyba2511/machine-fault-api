from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "machine_data.csv"
MODEL_PATH = BASE_DIR / "models" / "fault_model.joblib"

FEATURES = [
    "temperature",
    "pressure",
    "vibration",
    "run_hours",
    "alarm_count",
    "maintenance_days",
]
TARGET = "fault"


def train() -> None:
    df = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    X = df[FEATURES]
    y = df[TARGET]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Training completed with {len(df)} samples")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
