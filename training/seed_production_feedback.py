"""Seed labeled production-like feedback through the real model runtime.

This script is for local/demo learning only. It uses the current runtime model
for probabilities/predictions, then simulates later ground truth and persists
both into PostgreSQL. Rows use a DEMO-SEED- prefix so the script is idempotent.
"""

from uuid import uuid4

import numpy as np
from sqlalchemy import text

from app.database import get_engine
from app.ml import predict_fault, runtime
from app.repository import save_prediction, set_actual_fault
from app.schemas import PredictionRequest

SEED = 20261021
ROW_COUNT = 50
PREFIX = "DEMO-SEED-"


def _actual_fault_probability(
    temperature: float,
    vibration: float,
    run_hours: float,
    alarm_count: int,
    maintenance_days: int,
) -> float:
    # Deliberately related to the features, but noisy enough to create TP/TN/FP/FN.
    z = (
        -5.8
        + 0.045 * (temperature - 60)
        + 0.55 * (vibration - 2.0)
        + 0.00018 * run_hours
        + 0.30 * alarm_count
        + 0.007 * maintenance_days
    )
    return float(1.0 / (1.0 + np.exp(-z)))


def main() -> None:
    rng = np.random.default_rng(SEED)

    with get_engine().connect() as connection:
        existing = int(connection.execute(
            text("SELECT COUNT(*) FROM ml_prediction_history WHERE machine_id LIKE :prefix"),
            {"prefix": f"{PREFIX}%"},
        ).scalar_one())

    if existing:
        print(f"Demo feedback already exists: {existing} rows. Nothing inserted.")
        return

    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    actual_faults = 0

    for i in range(1, ROW_COUNT + 1):
        temperature = float(np.clip(rng.normal(68, 10), 45, 100))
        pressure = float(np.clip(rng.normal(5.2, 1.0), 2.0, 8.5))
        vibration = float(np.clip(rng.normal(2.5, 0.9), 0.5, 5.5))
        run_hours = float(np.clip(rng.normal(5000, 2200), 200, 12000))
        alarm_count = int(np.clip(rng.poisson(1.8), 0, 8))
        maintenance_days = int(np.clip(rng.normal(100, 60), 1, 365))

        request = PredictionRequest(
            machine_id=f"{PREFIX}{i:03d}",
            work_center="WC-DEMO",
            temperature=temperature,
            pressure=pressure,
            vibration=vibration,
            run_hours=run_hours,
            alarm_count=alarm_count,
            maintenance_days=maintenance_days,
        )

        result = predict_fault(request)
        prediction_id = save_prediction(uuid4(), request, result, runtime.current().threshold)

        p_actual = _actual_fault_probability(
            temperature, vibration, run_hours, alarm_count, maintenance_days
        )
        actual_fault = int(rng.random() < p_actual)
        set_actual_fault(prediction_id, actual_fault)
        actual_faults += actual_fault

        prediction = int(result["prediction"])
        if prediction == 1 and actual_fault == 1:
            counts["tp"] += 1
        elif prediction == 0 and actual_fault == 0:
            counts["tn"] += 1
        elif prediction == 1 and actual_fault == 0:
            counts["fp"] += 1
        else:
            counts["fn"] += 1

    loaded = runtime.current()
    print(f"Inserted {ROW_COUNT} labeled demo production rows")
    print(f"Runtime model: {loaded.version}")
    print(f"Runtime threshold: {loaded.threshold:.2f}")
    print(f"Actual faults: {actual_faults}/{ROW_COUNT}")
    print(f"TP={counts['tp']} TN={counts['tn']} FP={counts['fp']} FN={counts['fn']}")
    print("These rows are synthetic demo feedback, not real production evidence.")


if __name__ == "__main__":
    main()
