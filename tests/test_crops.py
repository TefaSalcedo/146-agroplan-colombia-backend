"""Tests for crop catalog endpoints."""


def test_list_crops(client):
    """List endpoint returns exactly 7 ML-supported crops."""
    response = client.get("/api/v1/crops")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 7
    crop_ids = {c["id"] for c in data["crops"]}
    expected = {"aguacate", "algodon", "cana_panelera", "cebolla", "fresa", "pina", "soya"}
    assert crop_ids == expected


def test_list_crops_lite(client):
    """Lite endpoint returns lightweight crop entries."""
    response = client.get("/api/v1/crops/lite")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 7
    for crop in data:
        assert "id" in crop
        assert "name" in crop
        assert "image" in crop
        assert "recommendation" in crop
        assert "success_rate" in crop


def test_get_crop_by_id(client):
    """Get crop by ID returns full details."""
    response = client.get("/api/v1/crops/aguacate")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "aguacate"
    assert data["name"] == "Aguacate"
    assert data["scientific_name"] == "Persea americana"
    assert data["days_to_harvest"] == 365


def test_get_crop_not_found(client):
    """Non-existent crop returns 404."""
    response = client.get("/api/v1/crops/cafe")
    assert response.status_code == 404
