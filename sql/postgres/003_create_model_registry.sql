CREATE TABLE IF NOT EXISTS ml_model_registry (
    id BIGSERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL UNIQUE,
    artifact_path VARCHAR(500) NOT NULL,
    status VARCHAR(20) NOT NULL,
    threshold NUMERIC(5,4) NOT NULL,
    validation_recall NUMERIC(9,8),
    validation_precision NUMERIC(9,8),
    validation_accuracy NUMERIC(9,8),
    training_rows INTEGER NOT NULL,
    trained_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    promoted_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ,
    CONSTRAINT ck_ml_model_registry_status
        CHECK (status IN ('CANDIDATE', 'PRODUCTION', 'RETIRED'))
);

-- PostgreSQL partial unique index: at most one PRODUCTION model.
CREATE UNIQUE INDEX IF NOT EXISTS ux_ml_model_registry_one_production
    ON ml_model_registry ((status))
    WHERE status = 'PRODUCTION';
