IF OBJECT_ID('dbo.ml_model_registry', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ml_model_registry (
        id BIGINT IDENTITY(1,1) PRIMARY KEY,
        model_version VARCHAR(50) NOT NULL,
        artifact_path NVARCHAR(500) NOT NULL,
        status VARCHAR(20) NOT NULL,
        threshold DECIMAL(5,4) NOT NULL,
        validation_recall DECIMAL(9,8) NULL,
        validation_precision DECIMAL(9,8) NULL,
        validation_accuracy DECIMAL(9,8) NULL,
        training_rows INT NOT NULL,
        trained_at DATETIME2(3) NOT NULL CONSTRAINT DF_ml_model_registry_trained_at DEFAULT SYSUTCDATETIME(),
        promoted_at DATETIME2(3) NULL,
        retired_at DATETIME2(3) NULL,
        CONSTRAINT UQ_ml_model_registry_version UNIQUE (model_version),
        CONSTRAINT CK_ml_model_registry_status CHECK (status IN ('CANDIDATE', 'PRODUCTION', 'RETIRED'))
    );

    CREATE UNIQUE INDEX UX_ml_model_registry_one_production
        ON dbo.ml_model_registry(status)
        WHERE status = 'PRODUCTION';
END;
