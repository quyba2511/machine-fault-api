"""Compare the registry PRODUCTION model with a CANDIDATE on the same labeled feedback.

This is an evaluation/demo tool. It does not promote or reload any model.
Both models receive exactly the same rows so their metrics and business cost are comparable.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sqlalchemy import text

from app.database import get_engine

FEATURES = ["temperature", "pressure", "vibration", "run_hours", "alarm_count", "maintenance_days"]
FN_COST = 10_000
FP_COST = 100


def get_registry_model(status: str, version: str | None = None) -> dict:
    clauses = ["status = :status"]
    params = {"status": status}
    if version:
        clauses.append("model_version = :version")
        params["version"] = version

    query = text(f"""
        SELECT model_version, artifact_path, threshold, status
        FROM ml_model_registry
        WHERE {' AND '.join(clauses)}
        ORDER BY promoted_at DESC NULLS LAST, id DESC
        LIMIT 1
    """)
    with get_engine().connect() as connection:
        row = connection.execute(query, params).mappings().first()
    if not row:
        label = version or status
        raise RuntimeError(f"No registry model found for {label}")
    return dict(row)


def load_feedback() -> pd.DataFrame:
    query = text("""
        SELECT id, created_at, temperature, pressure, vibration, run_hours,
               alarm_count, maintenance_days, actual_fault
        FROM ml_prediction_history
        WHERE actual_fault IS NOT NULL
        ORDER BY created_at, id
    """)
    return pd.read_sql(query, get_engine())


def evaluate(entry: dict, df: pd.DataFrame) -> dict:
    artifact = Path(entry["artifact_path"])
    if not artifact.exists():
        raise RuntimeError(f"Artifact does not exist for {entry['model_version']}: {artifact}")

    loaded = joblib.load(artifact)
    # v1.3 artifact may be a metadata wrapper while retrained candidates are pipeline-only.
    model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
    probability = model.predict_proba(df[FEATURES])[:, 1]
    threshold = float(entry["threshold"])
    prediction = (probability >= threshold).astype(int)
    actual = df["actual_fault"].astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, prediction, labels=[0, 1]).ravel()

    return {
        "model_version": entry["model_version"],
        "threshold": threshold,
        "rows": len(df),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy_score(actual, prediction)),
        "precision": float(precision_score(actual, prediction, zero_division=0)),
        "recall": float(recall_score(actual, prediction, zero_division=0)),
        "business_cost": int(fn * FN_COST + fp * FP_COST),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="v1.4")
    args = parser.parse_args()

    production = get_registry_model("PRODUCTION")
    candidate = get_registry_model("CANDIDATE", args.candidate)
    feedback = load_feedback()
    if feedback.empty:
        raise RuntimeError("No labeled production feedback is available")
    if feedback["actual_fault"].astype(int).nunique() < 2:
        raise RuntimeError("Comparison requires both fault and non-fault labeled rows")

    results = {
        "warning": "Workflow/demo comparison only; the current feedback set is small and includes synthetic demo labels.",
        "costs": {"false_negative": FN_COST, "false_positive": FP_COST},
        "evaluation_rows": len(feedback),
        "production": evaluate(production, feedback),
        "candidate": evaluate(candidate, feedback),
    }
    print(json.dumps(results, indent=2))
    print("\nNo model was promoted. This command only compares both models on the same labeled rows.")


if __name__ == "__main__":
    main()
