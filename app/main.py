from uuid import uuid4

from fastapi import FastAPI, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.ml import MODEL_VERSION, THRESHOLD, predict_fault
from app.repository import save_prediction, set_actual_fault
from app.schemas import ActualFaultUpdate, PredictionRequest, PredictionResponse

app = FastAPI(
    title="Machine Fault Prediction API",
    version="2.0.0",
    description="ML inference service with optional SQL Server prediction history.",
)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "prediction_persistence": settings.persist_predictions,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest) -> dict:
    request_id = uuid4()
    try:
        result = predict_fault(data)
        prediction_id = None

        if get_settings().persist_predictions:
            prediction_id = save_prediction(request_id, data, result, THRESHOLD)

        return {
            "request_id": request_id,
            "prediction_id": prediction_id,
            **result,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Prediction succeeded but persistence failed; transaction rolled back.",
        ) from exc


@app.patch("/predictions/{prediction_id}/actual-fault")
def update_actual_fault(prediction_id: int, payload: ActualFaultUpdate) -> dict:
    try:
        updated = set_actual_fault(prediction_id, payload.actual_fault)
    except (RuntimeError, SQLAlchemyError) as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc

    if not updated:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return {
        "prediction_id": prediction_id,
        "actual_fault": payload.actual_fault,
        "status": "updated",
    }
