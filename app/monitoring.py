from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from app.database import get_engine


PERFORMANCE_SQL = """
SELECT
    COUNT(*) AS labeled_count,
    SUM(CASE WHEN prediction IS TRUE AND actual_fault IS TRUE THEN 1 ELSE 0 END) AS tp,
    SUM(CASE WHEN prediction IS FALSE AND actual_fault IS FALSE THEN 1 ELSE 0 END) AS tn,
    SUM(CASE WHEN prediction IS TRUE AND actual_fault IS FALSE THEN 1 ELSE 0 END) AS fp,
    SUM(CASE WHEN prediction IS FALSE AND actual_fault IS TRUE THEN 1 ELSE 0 END) AS fn,
    AVG(CAST(fault_probability AS DOUBLE PRECISION)) AS avg_probability,
    AVG(CAST(CAST(actual_fault AS INTEGER) AS DOUBLE PRECISION)) AS actual_fault_rate
FROM ml_prediction_history
WHERE actual_fault IS NOT NULL
  AND created_at >= :since
"""

FEATURE_STATS_SQL = """
SELECT
    COUNT(*) AS prediction_count,
    AVG(temperature) AS temperature_mean,
    AVG(pressure) AS pressure_mean,
    AVG(vibration) AS vibration_mean,
    AVG(run_hours) AS run_hours_mean,
    AVG(CAST(alarm_count AS DOUBLE PRECISION)) AS alarm_count_mean,
    AVG(CAST(maintenance_days AS DOUBLE PRECISION)) AS maintenance_days_mean,
    AVG(CAST(fault_probability AS DOUBLE PRECISION)) AS probability_mean
FROM ml_prediction_history
WHERE created_at >= :since
"""


def _query_with_optional_model_version(base_sql: str, model_version: str | None):
    if model_version is None:
        return text(base_sql)
    return text(base_sql + "\n  AND model_version = :model_version")


def _query_params(since: datetime, model_version: str | None) -> dict:
    params = {"since": since}
    if model_version is not None:
        params["model_version"] = model_version
    return params


def _safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def get_model_performance(days: int = 30, model_version: str | None = None) -> dict:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    query = _query_with_optional_model_version(PERFORMANCE_SQL, model_version)
    params = _query_params(since, model_version)

    with get_engine().connect() as connection:
        row = connection.execute(query, params).mappings().one()

    tp = int(row["tp"] or 0)
    tn = int(row["tn"] or 0)
    fp = int(row["fp"] or 0)
    fn = int(row["fn"] or 0)
    total = int(row["labeled_count"] or 0)

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)

    return {
        "window_days": days,
        "model_version": model_version,
        "labeled_count": total,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": _safe_divide(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "actual_fault_rate": round(float(row["actual_fault_rate"]), 4) if row["actual_fault_rate"] is not None else None,
        "average_probability": round(float(row["avg_probability"]), 4) if row["avg_probability"] is not None else None,
    }


def get_feature_stats(days: int = 7, model_version: str | None = None) -> dict:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    query = _query_with_optional_model_version(FEATURE_STATS_SQL, model_version)
    params = _query_params(since, model_version)

    with get_engine().connect() as connection:
        row = connection.execute(query, params).mappings().one()

    return {
        "window_days": days,
        "model_version": model_version,
        "prediction_count": int(row["prediction_count"] or 0),
        "means": {
            key: round(float(row[key]), 4) if row[key] is not None else None
            for key in (
                "temperature_mean",
                "pressure_mean",
                "vibration_mean",
                "run_hours_mean",
                "alarm_count_mean",
                "maintenance_days_mean",
                "probability_mean",
            )
        },
    }
