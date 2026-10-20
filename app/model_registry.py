from sqlalchemy import text

from app.database import get_engine


GET_PRODUCTION = text("""
SELECT TOP 1 model_version, artifact_path, threshold
FROM dbo.ml_model_registry
WHERE status = 'PRODUCTION'
ORDER BY promoted_at DESC
""")

PROMOTE = text("""
UPDATE dbo.ml_model_registry
SET status = 'RETIRED', retired_at = SYSUTCDATETIME()
WHERE status = 'PRODUCTION';

UPDATE dbo.ml_model_registry
SET status = 'PRODUCTION', promoted_at = SYSUTCDATETIME(), retired_at = NULL
WHERE model_version = :model_version AND status IN ('CANDIDATE', 'RETIRED');
""")


def get_production_model() -> dict | None:
    with get_engine().connect() as connection:
        row = connection.execute(GET_PRODUCTION).mappings().first()
    return dict(row) if row else None


def promote_model(model_version: str) -> bool:
    # Both retirement of the current model and promotion of the candidate
    # happen in one transaction. If promotion fails, the old production model
    # remains production after rollback.
    with get_engine().begin() as connection:
        candidate = connection.execute(
            text("SELECT status FROM dbo.ml_model_registry WITH (UPDLOCK, HOLDLOCK) WHERE model_version = :v"),
            {"v": model_version},
        ).scalar_one_or_none()
        if candidate not in ("CANDIDATE", "RETIRED"):
            return False
        connection.execute(PROMOTE, {"model_version": model_version})
    return True
