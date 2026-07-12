"""Tests for admin endpoints."""


def test_admin_climate_sync_status_no_key_required(client):
    """Climate sync status does not require admin API key."""
    response = client.get("/api/v1/admin/climate-sync/status")
    assert response.status_code == 200
    data = response.json()
    assert "forecast_records" in data
    assert "municipalities_with_forecast" in data


def test_admin_models_status_without_key(client):
    """Model status endpoint requires admin API key."""
    response = client.get("/api/v1/admin/models/status")
    assert response.status_code == 401


def test_admin_models_status_with_key(client, admin_headers):
    """Model status endpoint returns model info with valid key."""
    response = client.get("/api/v1/admin/models/status", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()
    assert "models_loaded" in data
    assert "zoning_models" in data
    assert "yield_models" in data
    assert "profiles_loaded" in data


def test_admin_cache_stats_without_key(client):
    """Cache stats endpoint requires admin API key."""
    response = client.get("/api/v1/admin/cache/stats")
    assert response.status_code == 401


def test_admin_cache_stats_with_key(client, admin_headers):
    """Cache stats endpoint returns cache counts with valid key."""
    response = client.get("/api/v1/admin/cache/stats", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()
    assert "total_entries" in data
    assert "active_entries" in data
    assert "expired_entries" in data
    assert "by_type" in data


def test_admin_cache_invalidate(client, admin_headers):
    """Cache invalidation works with admin key."""
    response = client.post(
        "/api/v1/admin/cache/invalidate",
        json={"prediction_type": "zoning"},
        headers=admin_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "invalidated" in data
    assert data["invalidated"] >= 0


def test_admin_audit_recent(client, admin_headers):
    """Audit endpoint returns recent prediction runs."""
    response = client.get("/api/v1/admin/audit/recent", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
