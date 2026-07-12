"""Tests for calendar endpoints."""


def test_calendar_batch(client):
    """Batch calendar returns 12-month predictions for specified crops."""
    response = client.post(
        "/api/v1/calendars/predict-batch",
        json={
            "municipality_id": "05001",
            "crop_ids": ["aguacate", "pina"],
            "horizon_months": 12,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["municipality_id"] == "05001"
    assert data["municipality_name"] == "MEDELL\u00cdN"
    assert data["horizon_months"] == 12
    assert len(data["results"]) == 2

    for result in data["results"]:
        assert "crop_id" in result
        assert "crop_name" in result
        assert "top_harvest_months" in result
        assert "monthly_forecasts" in result
        assert "warnings" in result
        assert "method" in result
        assert len(result["monthly_forecasts"]) == 12

        # Each monthly forecast should have climate_source
        for mf in result["monthly_forecasts"]:
            assert mf["climate_source"] in (
                "open_meteo_forecast",
                "historical_climatology",
                "not_available",
            )

        # Top harvest months should have at most 3 entries
        assert len(result["top_harvest_months"]) <= 3

        # Each harvest window should have month name and score
        for hw in result["top_harvest_months"]:
            assert "harvest_month" in hw
            assert "harvest_month_name" in hw
            assert "score" in hw
            assert 0.0 <= hw["score"] <= 1.0


def test_calendar_batch_all_crops(client):
    """Batch calendar without crop_ids returns all ML-supported crops."""
    response = client.post(
        "/api/v1/calendars/predict-batch",
        json={"municipality_id": "05001"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 7


def test_calendar_batch_municipality_not_found(client):
    """Non-existent municipality returns 404."""
    response = client.post(
        "/api/v1/calendars/predict-batch",
        json={"municipality_id": "99999"},
    )
    assert response.status_code == 404


def test_calendar_batch_explanation_per_crop(client):
    """Each crop result includes its own LLM explanation audit fields."""
    response = client.post(
        "/api/v1/calendars/predict-batch",
        json={"municipality_id": "05001", "crop_ids": ["aguacate"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert "explanation" not in data
    for result in data["results"]:
        assert "explanation" in result
        explanation = result["explanation"]
        assert "status" in explanation
        assert explanation["status"] in ("success", "llm_unavailable")
        assert "tokens_in" in explanation
        assert "tokens_out" in explanation
        assert "tokens_total" in explanation
        assert "provider" in explanation
        assert "model" in explanation
