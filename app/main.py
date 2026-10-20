from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.database import database_is_ready
from app.deployment import reload_production_model
from app.logging_config import configure_logging
from app.middleware import request_logging_middleware
from app.ml import get_runtime_info, load_model, predict_fault, runtime
from app.monitoring import get_feature_stats, get_model_performance
from app.repository import save_prediction, set_actual_fault
from app.schemas import ActualFaultUpdate, PredictionRequest, PredictionResponse

configure_logging()

app = FastAPI(title="Machine Fault Prediction API", version="6.0.0", description="Production ML service with runtime model lifecycle.")
app.middleware("http")(request_logging_middleware)


@app.get("/health/live")
def liveness() -> dict:
    return {"status": "ok"}


@app.get("/health/ready")
def readiness() -> dict:
    settings = get_settings()
    checks = {"model": True, "database": None}
    try:
        load_model()
        model_info = get_runtime_info()
    except RuntimeError:
        checks["model"] = False
        model_info = None
    if settings.persist_predictions:
        checks["database"] = database_is_ready()
    ready = checks["model"] and checks["database"] is not False
    if not ready:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})
    return {"status": "ready", "checks": checks, "runtime": model_info}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "runtime": get_runtime_info(), "prediction_persistence": get_settings().persist_predictions}


@app.get("/models/runtime")
def runtime_model() -> dict:
    return get_runtime_info()


@app.post("/models/reload-production")
def reload_model() -> dict:
    try:
        return reload_production_model()
    except (RuntimeError, SQLAlchemyError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest, request: Request) -> dict:
    header_request_id = getattr(request.state, "request_id", None)
    try:
        request_id = UUID(header_request_id) if header_request_id else uuid4()
    except ValueError:
        request_id = uuid4()
    try:
        result = predict_fault(data)
        prediction_id = None
        if get_settings().persist_predictions:
            prediction_id = save_prediction(request_id, data, result, runtime.current().threshold)
        return {"request_id": request_id, "prediction_id": prediction_id, **result}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Prediction succeeded but persistence failed; transaction rolled back.") from exc


@app.patch("/predictions/{prediction_id}/actual-fault")
def update_actual_fault(prediction_id: int, payload: ActualFaultUpdate) -> dict:
    try:
        updated = set_actual_fault(prediction_id, payload.actual_fault)
    except (RuntimeError, SQLAlchemyError) as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    if not updated:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"prediction_id": prediction_id, "actual_fault": payload.actual_fault, "status": "updated"}


@app.get("/monitoring/performance")
def monitoring_performance(days: int = Query(default=30, ge=1, le=365), model_version: str | None = None) -> dict:
    try:
        return get_model_performance(days, model_version)
    except (RuntimeError, SQLAlchemyError) as exc:
        raise HTTPException(status_code=503, detail="Monitoring database unavailable") from exc


@app.get("/monitoring/features")
def monitoring_features(days: int = Query(default=7, ge=1, le=365), model_version: str | None = None) -> dict:
    try:
        return get_feature_stats(days, model_version)
    except (RuntimeError, SQLAlchemyError) as exc:
        raise HTTPException(status_code=503, detail="Monitoring database unavailable") from exc
