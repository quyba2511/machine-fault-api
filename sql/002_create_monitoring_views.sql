CREATE OR ALTER VIEW dbo.v_ml_daily_monitoring
AS
SELECT
    CAST(created_at AS DATE) AS prediction_date,
    model_version,
    COUNT(*) AS prediction_count,
    AVG(CAST(fault_probability AS FLOAT)) AS avg_fault_probability,
    SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) AS predicted_fault_count,
    SUM(CASE WHEN actual_fault = 1 THEN 1 ELSE 0 END) AS actual_fault_count,
    SUM(CASE WHEN actual_fault IS NOT NULL THEN 1 ELSE 0 END) AS labeled_count,
    SUM(CASE WHEN prediction = actual_fault AND actual_fault IS NOT NULL THEN 1 ELSE 0 END) AS correct_count,
    AVG(temperature) AS avg_temperature,
    AVG(pressure) AS avg_pressure,
    AVG(vibration) AS avg_vibration,
    AVG(run_hours) AS avg_run_hours,
    AVG(CAST(alarm_count AS FLOAT)) AS avg_alarm_count,
    AVG(CAST(maintenance_days AS FLOAT)) AS avg_maintenance_days
FROM dbo.ml_prediction_history
GROUP BY CAST(created_at AS DATE), model_version;
