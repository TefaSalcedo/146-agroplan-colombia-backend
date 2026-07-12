import secrets

from fastapi import Header, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.config import get_settings

__all__ = ["get_db", "get_settings", "verify_admin_api_key"]


def verify_admin_api_key(
    x_admin_api_key: str = Header(default="", alias="X-Admin-API-Key"),
) -> None:
    """Verify the admin API key using constant-time comparison."""
    settings = get_settings()
    expected = settings.admin_api_key

    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API key is not configured on the server",
        )

    if not secrets.compare_digest(x_admin_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key",
        )
