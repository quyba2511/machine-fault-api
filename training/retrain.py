import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from app.database import get_engine

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES = ["temperature", "pressure", "vibration", "run_hours", "alarm_count", "maintenance_days"]
MIN_LABELED_ROWS = 30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Candidate version, e.g. v1.4")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    query = text("""
        SELECT temperature, pressure, vibration, run_hours, alarm_count,
               maintenance_days, actual_fault
        FROM ml_prediction_history
        WHERE actual_fault IS NOT NULL
        ORDER BY created_at
    """)
    df = pd.read_sql(query, get_engine())
    if len(df) < MIN_LABELED_ROWS:
        raise RuntimeError(
            f"At least {MIN_LABELED_ROWS} labeled production rows are required for retraining; "
            f"found {len(df)}"
        )

    class_counts = df["actual_fault"].astype(int).value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        raise RuntimeError("Retraining requires at least two labeled rows in each class")

    X_train, X_valid, y_train, y_valid = train_test_split(
        df[FEATURES],
        df["actual_fault"].astype(int),
        test_size=0.25,
        random_state=42,
        stratify=df["actual_fault"].astype(int),
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    probability = pipeline.predict_proba(X_valid)[:, 1]
    prediction = (probability >= args.threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_valid, prediction)),
        "precision": float(precision_score(y_valid, prediction, zero_division=0)),
        "recall": float(recall_score(y_valid, prediction, zero_division=0)),
    }

    artifact_dir = BASE_DIR / "models" / args.version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "fault_model.joblib"
    metadata_path = artifact_dir / "metadata.json"
    joblib.dump(pipeline, artifact_path)
    metadata_path.write_text(json.dumps({
        "model_version": args.version,
        "threshold": args.threshold,
        "training_rows": len(X_train),
        "validation_rows": len(X_valid),
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")

    with get_engine().begin() as connection:
        connection.execute(text("""
            INSERT INTO ml_model_registry (
                model_version, artifact_path, status, threshold,
                validation_recall, validation_precision, validation_accuracy, training_rows
            ) VALUES (
                :version, :path, 'CANDIDATE', :threshold,
                :recall, :precision, :accuracy, :training_rows
            )
        """), {
            "version": args.version,
            "path": str(artifact_path),
            "threshold": args.threshold,
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "accuracy": metrics["accuracy"],
            "training_rows": len(X_train),
        })

    print(json.dumps({
        "labeled_rows": len(df),
        "training_rows": len(X_train),
        "validation_rows": len(X_valid),
        "metrics": metrics,
    }, indent=2))
    print(f"Registered {args.version} as CANDIDATE")


if __name__ == "__main__":
    main()
