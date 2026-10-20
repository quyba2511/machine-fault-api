IF OBJECT_ID('dbo.ml_prediction_history', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ml_prediction_history (
        id BIGINT IDENTITY(1,1) PRIMARY KEY,
        request_id UNIQUEIDENTIFIER NOT NULL,
        machine_id NVARCHAR(50) NOT NULL,
        temperature FLOAT NOT NULL,
        pressure FLOAT NOT NULL,
        vibration FLOAT NOT NULL,
        run_hours FLOAT NOT NULL,
        alarm_count INT NOT NULL,
        maintenance_days INT NOT NULL,
        fault_probability DECIMAL(9,8) NOT NULL,
        prediction BIT NOT NULL,
        risk_level VARCHAR(10) NOT NULL,
        threshold DECIMAL(5,4) NOT NULL,
        model_version VARCHAR(50) NOT NULL,
        actual_fault BIT NULL,
        created_at DATETIME2(3) NOT NULL CONSTRAINT DF_ml_prediction_history_created_at DEFAULT SYSUTCDATETIME(),
        actual_fault_updated_at DATETIME2(3) NULL,
        CONSTRAINT UQ_ml_prediction_history_request_id UNIQUE (request_id),
        CONSTRAINT CK_ml_prediction_history_probability CHECK (fault_probability >= 0 AND fault_probability <= 1),
        CONSTRAINT CK_ml_prediction_history_risk CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH'))
    );

    CREATE INDEX IX_ml_prediction_history_machine_created
        ON dbo.ml_prediction_history(machine_id, created_at DESC);

    CREATE INDEX IX_ml_prediction_history_model_created
        ON dbo.ml_prediction_history(model_version, created_at DESC);
END;
