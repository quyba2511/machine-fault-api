import hmac

from fastapi import Header, HTTPException

from app.config import get_settings


def require_admin_api_key(x_admin_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().admin_api_key
    if not expected:
        raise HTTPException(status_code=503, detail="Admin authentication is not configured")
    if not x_admin_api_key or not hmac.compare_digest(x_admin_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
