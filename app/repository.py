from uuid import UUID

from sqlalchemy import text

from app.database import get_engine
from app.schemas import PredictionRequest


INSERT_PREDICTION = text("""
INSERT INTO ml_prediction_history (
    request_id, machine_id, temperature, pressure, vibration,
    run_hours, alarm_count, maintenance_days,
    fault_probability, prediction, risk_level, threshold, model_version
)
VALUES (
    :request_id, :machine_id, :temperature, :pressure, :vibration,
    :run_hours, :alarm_count, :maintenance_days,
    :fault_probability, :prediction, :risk_level, :threshold, :model_version
)
RETURNING id
""")

UPDATE_ACTUAL_FAULT = text("""
UPDATE ml_prediction_history
SET actual_fault = :actual_fault,
    actual_fault_updated_at = CURRENT_TIMESTAMP
WHERE id = :prediction_id
""")


def save_prediction(
    request_id: UUID,
    request: PredictionRequest,
    result: dict,
    threshold: float,
) -> int:
    params = {
        "request_id": str(request_id),
        "machine_id": request.machine_id,
        "temperature": request.temperature,
        "pressure": request.pressure,
        "vibration": request.vibration,
        "run_hours": request.run_hours,
        "alarm_count": request.alarm_count,
        "maintenance_days": request.maintenance_days,
        "fault_probability": result["fault_probability"],
        "prediction": bool(result["prediction"]),
        "risk_level": result["risk_level"],
        "threshold": threshold,
        "model_version": result["model_version"],
    }

    # engine.begin() commits on success and rolls back automatically on exception.
    with get_engine().begin() as connection:
        prediction_id = connection.execute(INSERT_PREDICTION, params).scalar_one()

    return int(prediction_id)


def set_actual_fault(prediction_id: int, actual_fault: int) -> bool:
    with get_engine().begin() as connection:
        result = connection.execute(
            UPDATE_ACTUAL_FAULT,
            {"prediction_id": prediction_id, "actual_fault": bool(actual_fault)},
        )
    return result.rowcount == 1
