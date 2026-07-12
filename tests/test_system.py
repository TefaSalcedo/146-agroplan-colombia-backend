"""Tests for system endpoints (root, health, readiness)."""


def test_root(client):
    """Root endpoint returns service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "AgroPlan Colombia Backend API"
    assert data["version"] == "2.0.0"
    assert data["docs"] == "/docs"


def test_health(client):
    """Health endpoint returns status and model info."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "2.0.0"
    assert "models_loaded" in data
    assert isinstance(data["models_loaded"], bool)


def test_readiness(client):
    """Readiness endpoint returns component-level status."""
    response = client.get("/api/v1/readiness")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "2.0.0"
    assert "status" in data
    assert "components" in data
    component_names = [c["name"] for c in data["components"]]
    assert "database" in component_names
    assert "zoning_model" in component_names
    assert "yield_models" in component_names
