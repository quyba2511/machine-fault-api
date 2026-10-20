from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_invalid_negative_run_hours():
    response = client.post(
        "/predict",
        json={
            "machine_id": "MC001",
            "temperature": 85,
            "pressure": 5.1,
            "vibration": 6.8,
            "run_hours": -1,
            "alarm_count": 5,
            "maintenance_days": 90,
        },
    )
    assert response.status_code == 422


def test_predict_after_training():
    response = client.post(
        "/predict",
        json={
            "machine_id": "MC001",
            "temperature": 85,
            "pressure": 5.1,
            "vibration": 6.8,
            "run_hours": 5200,
            "alarm_count": 5,
            "maintenance_days": 90,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["machine_id"] == "MC001"
    assert 0 <= body["fault_probability"] <= 1
    assert body["prediction"] in (0, 1)
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")
