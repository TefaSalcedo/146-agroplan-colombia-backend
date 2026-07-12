"""Tests for zoning endpoints."""


def test_zoning_predict(client):
    """Single-crop zoning prediction returns suitability and method."""
    response = client.post(
        "/api/v1/zoning/predict",
        json={"crop_id": "aguacate", "municipality_id": "05001"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["crop_id"] == "aguacate"
    assert data["municipality_id"] == "05001"
    assert data["suitability"] in ("high", "medium", "low", "none")
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["model_version"] == "mock-v1"
    assert "method" in data
    assert "cache_hit" in data
    assert "factors" in data
    factors = data["factors"]
    assert "temperature_match" in factors
    assert "precipitation_match" in factors
    assert "soil_match" in factors
    assert "altitude_match" in factors


def test_zoning_predict_cache_hit(client):
    """Second call to same zoning prediction returns cache hit."""
    payload = {"crop_id": "fresa", "municipality_id": "05001"}
    r1 = client.post("/api/v1/zoning/predict", json=payload)
    assert r1.status_code == 200
    assert r1.json()["cache_hit"] is False

    r2 = client.post("/api/v1/zoning/predict", json=payload)
    assert r2.status_code == 200
    assert r2.json()["cache_hit"] is True


def test_zoning_predict_municipality_not_found(client):
    """Non-existent municipality returns 404."""
    response = client.post(
        "/api/v1/zoning/predict",
        json={"crop_id": "aguacate", "municipality_id": "99999"},
    )
    assert response.status_code == 404


def test_zoning_predict_crop_not_found(client):
    """Non-existent crop returns 404."""
    response = client.post(
        "/api/v1/zoning/predict",
        json={"crop_id": "cafe", "municipality_id": "05001"},
    )
    assert response.status_code == 404


def test_zoning_recommendations(client):
    """Batch zoning returns all 7 crops ranked by confidence."""
    response = client.post(
        "/api/v1/zoning/recommendations",
        json={"municipality_id": "05001"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["municipality_id"] == "05001"
    assert data["municipality_name"] == "MEDELL\u00cdN"
    assert len(data["results"]) == 7
    # Results should be sorted by confidence descending
    confidences = [r["confidence"] for r in data["results"]]
    assert confidences == sorted(confidences, reverse=True)
    # Each result should have method field
    for r in data["results"]:
        assert r["method"] in ("primary_model", "fallback", "unavailable", "mock")


def test_zoning_recommendations_with_specific_crops(client):
    """Batch zoning with specific crop_ids returns only those crops."""
    response = client.post(
        "/api/v1/zoning/recommendations",
        json={"municipality_id": "05001", "crop_ids": ["aguacate", "pina"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    crop_ids = {r["crop_id"] for r in data["results"]}
    assert crop_ids == {"aguacate", "pina"}
