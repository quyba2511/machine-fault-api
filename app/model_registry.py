from sqlalchemy import text

from app.database import get_engine


GET_PRODUCTION = text("""
SELECT model_version, artifact_path, threshold
FROM ml_model_registry
WHERE status = 'PRODUCTION'
ORDER BY promoted_at DESC NULLS LAST, id DESC
LIMIT 1
""")


def get_production_model() -> dict | None:
    with get_engine().connect() as connection:
        row = connection.execute(GET_PRODUCTION).mappings().first()
    return dict(row) if row else None


def promote_model(model_version: str) -> bool:
    # PostgreSQL transaction + row locking keeps promotion atomic.
    # If the candidate cannot be promoted, the current production model is
    # left unchanged because the transaction is rolled back/never modified.
    with get_engine().begin() as connection:
        candidate = connection.execute(
            text("""
                SELECT status
                FROM ml_model_registry
                WHERE model_version = :v
                FOR UPDATE
            """),
            {"v": model_version},
        ).scalar_one_or_none()

        if candidate not in ("CANDIDATE", "RETIRED"):
            return False

        connection.execute(text("""
            UPDATE ml_model_registry
            SET status = 'RETIRED', retired_at = CURRENT_TIMESTAMP
            WHERE status = 'PRODUCTION'
        """))

        result = connection.execute(text("""
            UPDATE ml_model_registry
            SET status = 'PRODUCTION',
                promoted_at = CURRENT_TIMESTAMP,
                retired_at = NULL
            WHERE model_version = :v
              AND status IN ('CANDIDATE', 'RETIRED')
        """), {"v": model_version})

        return result.rowcount == 1
