from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

client = TestClient(app)


def test_admin_endpoint_rejects_missing_key(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "test-secret")
    get_settings.cache_clear()
    response = client.post("/models/reload-production")
    assert response.status_code == 401
    get_settings.cache_clear()


def test_admin_endpoint_rejects_wrong_key(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "test-secret")
    get_settings.cache_clear()
    response = client.post(
        "/models/reload-production",
        headers={"X-Admin-API-Key": "wrong-secret"},
    )
    assert response.status_code == 401
    get_settings.cache_clear()


def test_admin_endpoint_fails_closed_without_configuration(monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    get_settings.cache_clear()
    response = client.post("/models/reload-production")
    assert response.status_code == 503
    get_settings.cache_clear()
