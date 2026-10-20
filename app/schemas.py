from uuid import UUID

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    machine_id: str = Field(min_length=1, max_length=50)
    temperature: float
    pressure: float
    vibration: float
    run_hours: float = Field(ge=0)
    alarm_count: int = Field(ge=0)
    maintenance_days: int = Field(ge=0)


class PredictionResponse(BaseModel):
    request_id: UUID
    prediction_id: int | None = None
    machine_id: str
    fault_probability: float
    prediction: int
    risk_level: str
    model_version: str


class ActualFaultUpdate(BaseModel):
    actual_fault: int = Field(ge=0, le=1)
