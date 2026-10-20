from app.ml import runtime
from app.model_registry import get_production_model


def reload_production_model() -> dict:
    registered = get_production_model()
    if not registered:
        raise RuntimeError("No PRODUCTION model is registered")

    loaded = runtime.reload(
        artifact_path=registered["artifact_path"],
        version=registered["model_version"],
        threshold=float(registered["threshold"]),
    )
    return {
        "model_version": loaded.version,
        "threshold": loaded.threshold,
        "artifact_path": loaded.artifact_path,
        "status": "loaded",
    }
