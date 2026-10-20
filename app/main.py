from fastapi import FastAPI, HTTPException

from app.ml import MODEL_VERSION, predict_fault
from app.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Machine Fault Prediction API",
    version="1.0.0",
    description="ML inference service for machine fault risk.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest) -> dict:
    try:
        return predict_fault(data)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
