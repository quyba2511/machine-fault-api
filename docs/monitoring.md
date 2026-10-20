# ML monitoring

A deployed model is not finished when `/predict` works. Production monitoring closes the loop between predictions and later ground truth.

## Feedback loop

Prediction -> store input/output/model version -> actual outcome arrives -> update `actual_fault` -> compute production metrics -> investigate degradation -> retrain/release only when justified.

## Performance endpoint

`GET /monitoring/performance?days=30&model_version=v1.0`

Returns labeled sample count, TP/TN/FP/FN, accuracy, precision, recall, actual fault rate and average predicted probability.

Recall is especially useful when the business cost of missing a real machine fault is high. A threshold must still be selected from business cost, not from a universal ML rule.

## Feature monitoring

`GET /monitoring/features?days=7`

Returns recent feature means and prediction volume. This is an intentionally simple first drift signal. A change in feature statistics is evidence to investigate, not proof that a model is bad.

Later versions can store training reference distributions and calculate PSI, KS statistics, missing-value rates and category-frequency drift.

## Retraining gate

`python monitoring/check_model.py --days 30 --min-labeled 30 --min-recall 0.70`

The command exits with code 2 when enough labeled data exists and recall falls below the configured business threshold. It does not retrain automatically. In an enterprise system, degradation should first trigger review because bad labels, sensor changes, upstream schema changes or temporary operating conditions can look like model degradation.

## Daily SQL view

Run `sql/002_create_monitoring_views.sql`. The resulting `dbo.v_ml_daily_monitoring` view supports dashboards and SQL analysis by date and model version.
