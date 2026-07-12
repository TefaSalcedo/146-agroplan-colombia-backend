"""Test configuration and fixtures."""
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Set test environment variables before importing app
os.environ["DATABASE_URL"] = "postgresql://agroplan:agroplan@localhost:5432/agroplan"
os.environ["MIGRATION_DATABASE_URL"] = "postgresql://agroplan:agroplan@localhost:5432/agroplan"
os.environ["ENABLE_CLIMATE_SYNC"] = "false"
os.environ["ADMIN_API_KEY"] = "test-admin-key-12345"
os.environ["ML_MODELS_PATH"] = "./models"


@pytest.fixture(scope="session")
def client():
    """FastAPI test client with real database."""
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def db_session():
    """Database session for direct DB assertions."""
    from app.database import SessionLocal

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def admin_headers():
    """Headers with admin API key."""
    return {"X-Admin-API-Key": "test-admin-key-12345"}
