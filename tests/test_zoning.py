"""Tests for zoning endpoints."""


def test_zoning_recommendations(client):
    """Municipality-only zoning returns all supported crops ranked by confidence."""
    response = client.get("/api/v1/zoning/recommendations/05001")
    assert response.status_code == 200
    data = response.json()
    assert data["municipality_id"] == "05001"
    assert data["municipality_name"] == "MEDELL\u00cdN"
    assert len(data["results"]) == 7
    # Results should be sorted by confidence descending
    confidences = [r["confidence"] for r in data["results"]]
    assert confidences == sorted(confidences, reverse=True)
    # Each result should have a valid method
    for r in data["results"]:
        assert r["method"] in ("primary_model", "climate_analog", "unavailable", "mock")


def test_zoning_recommendations_municipality_not_found(client):
    """Non-existent municipality returns 404."""
    response = client.get("/api/v1/zoning/recommendations/99999")
    assert response.status_code == 404


def test_zoning_recommendations_include_climate_based(client):
    """Municipality-only zoning includes climate-based recommendations."""
    response = client.get("/api/v1/zoning/recommendations/05001")
    assert response.status_code == 200
    data = response.json()
    assert "climate_based_recommendations" in data
    # Each climate recommendation should have expected fields
    for rec in data["climate_based_recommendations"]:
        assert "crop_id" in rec
        assert "crop_name" in rec
        assert "score" in rec
        assert "source" in rec
        assert rec["source"] == "climate_analog_knn"


def test_zoning_map(client):
    """Zoning map returns one result per municipality for the requested crop."""
    response = client.get("/api/v1/zoning/map/aguacate")
    assert response.status_code == 200
    data = response.json()
    assert data["crop_id"] == "aguacate"
    assert data["crop_name"] == "Aguacate"
    assert data["total_municipalities"] > 0
    assert len(data["results"]) == data["total_municipalities"]
    first = data["results"][0]
    assert "municipality_id" in first
    assert "municipality_name" in first
    assert "suitability" in first
    assert "confidence" in first
    assert "method" in first


def test_zoning_map_crop_not_found(client):
    """Non-existent crop returns 404."""
    response = client.get("/api/v1/zoning/map/cafe")
    assert response.status_code == 404
