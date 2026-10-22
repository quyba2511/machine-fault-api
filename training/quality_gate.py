"""Evaluate a candidate against production on the candidate's unseen validation rows.

The current retraining code uses train_test_split(random_state=42, stratify=y).
This script reconstructs exactly that split from the same ordered feedback rows and
uses ONLY the validation indices. Therefore the candidate is not evaluated on the
rows used to fit it.

This is still a small/demo dataset. A future production version should persist the
split/evaluation cohort explicitly and preferably use a temporal holdout.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sqlalchemy import text

from app.database import get_engine

FEATURES = ["temperature", "pressure", "vibration", "run_hours", "alarm_count", "maintenance_days"]
FN_COST = 10_000
FP_COST = 100
MIN_RECALL = 0.75


def registry_model(status: str, version: str | None = None) -> dict:
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
        raise RuntimeError(f"No registry model found for {version or status}")
    return dict(row)


def feedback() -> pd.DataFrame:
    return pd.read_sql(text("""
        SELECT id, created_at, temperature, pressure, vibration, run_hours,
               alarm_count, maintenance_days, actual_fault
        FROM ml_prediction_history
        WHERE actual_fault IS NOT NULL
        ORDER BY created_at, id
    """), get_engine())


def unseen_validation(df: pd.DataFrame) -> pd.DataFrame:
    y = df["actual_fault"].astype(int)
    indices = df.index.to_numpy()
    _, valid_indices = train_test_split(
        indices,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    return df.loc[valid_indices].sort_values(["created_at", "id"]).copy()


def evaluate(entry: dict, df: pd.DataFrame) -> dict:
    artifact = Path(entry["artifact_path"])
    if not artifact.exists():
        raise RuntimeError(f"Artifact does not exist for {entry['model_version']}: {artifact}")
    loaded = joblib.load(artifact)
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
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "accuracy": float(accuracy_score(actual, prediction)),
        "precision": float(precision_score(actual, prediction, zero_division=0)),
        "recall": float(recall_score(actual, prediction, zero_division=0)),
        "business_cost": int(fn * FN_COST + fp * FP_COST),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="v1.4")
    args = parser.parse_args()

    df = feedback()
    if len(df) < 30:
        raise RuntimeError("At least 30 labeled rows are required")
    if df["actual_fault"].astype(int).value_counts().min() < 2:
        raise RuntimeError("Both classes need at least two rows")

    holdout = unseen_validation(df)
    production_result = evaluate(registry_model("PRODUCTION"), holdout)
    candidate_result = evaluate(registry_model("CANDIDATE", args.candidate), holdout)

    checks = {
        "candidate_recall_at_least_minimum": candidate_result["recall"] >= MIN_RECALL,
        "candidate_recall_not_worse_than_production": candidate_result["recall"] >= production_result["recall"],
        "candidate_business_cost_not_worse_than_production": candidate_result["business_cost"] <= production_result["business_cost"],
    }
    passed = all(checks.values())

    print(json.dumps({
        "warning": "Demo quality gate: only the candidate's reconstructed unseen validation rows are used; sample size is still very small.",
        "evaluation": "candidate_unseen_validation_only",
        "holdout_rows": len(holdout),
        "holdout_faults": int(holdout["actual_fault"].astype(int).sum()),
        "costs": {"false_negative": FN_COST, "false_positive": FP_COST},
        "minimum_recall": MIN_RECALL,
        "production": production_result,
        "candidate": candidate_result,
        "checks": checks,
        "quality_gate": "PASS" if passed else "REJECT",
    }, indent=2))
    print("\nNo model was promoted. Promotion remains a separate explicit action.")


if __name__ == "__main__":
    main()
