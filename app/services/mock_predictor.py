from datetime import datetime, timedelta
from typing import Dict

import random
from sqlalchemy.orm import Session

from app.models import MunicipalityClimateForecast
from app.services.municipality_catalog import MunicipalityCatalog

municipality_catalog = MunicipalityCatalog()


def get_municipality_climate_summary(db: Session, municipality_id: str) -> Dict:
    """Compute average climate values from stored forecasts for a municipality."""
    today = datetime.utcnow().date()
    future_cutoff = today + timedelta(days=90)

    records = (
        db.query(MunicipalityClimateForecast)
        .filter(MunicipalityClimateForecast.municipality_id == municipality_id)
        .filter(MunicipalityClimateForecast.forecast_date >= today)
        .filter(MunicipalityClimateForecast.forecast_date <= future_cutoff)
        .all()
    )

    if not records:
        return {"avg_temp": 0.0, "avg_precipitation": 0.0, "has_data": False}

    avg_temp = sum(r.temp_mean for r in records if r.temp_mean is not None) / max(
        1, sum(1 for r in records if r.temp_mean is not None)
    )
    avg_precipitation = sum(r.precipitation for r in records if r.precipitation is not None) / max(
        1, sum(1 for r in records if r.precipitation is not None)
    )

    return {
        "avg_temp": round(avg_temp, 2),
        "avg_precipitation": round(avg_precipitation, 2),
        "has_data": True,
    }


class MockPredictor:
    """Mock predictor for zoning and calendar predictions.

    This provides mock predictions based on simple rules while the real ML models
    are being trained. When models are available, this will be replaced by model_loader.py.
    """

    _SUITABILITY_RULES = {
        "cafe": {
            "min_alt": 1200,
            "max_alt": 2000,
            "min_temp": 18,
            "max_temp": 24,
        },
        "maiz": {
            "min_alt": 0,
            "max_alt": 2800,
            "min_temp": 20,
            "max_temp": 30,
        },
        "frijol": {
            "min_alt": 1000,
            "max_alt": 2400,
            "min_temp": 15,
            "max_temp": 27,
        },
        "aguacate": {
            "min_alt": 1500,
            "max_alt": 2500,
            "min_temp": 16,
            "max_temp": 24,
        },
        "papa": {
            "min_alt": 2000,
            "max_alt": 3200,
            "min_temp": 14,
            "max_temp": 18,
        },
        "tomate": {
            "min_alt": 0,
            "max_alt": 2000,
            "min_temp": 18,
            "max_temp": 25,
        },
        "platano": {
            "min_alt": 0,
            "max_alt": 1500,
            "min_temp": 25,
            "max_temp": 30,
        },
        "cacao": {
            "min_alt": 0,
            "max_alt": 800,
            "min_temp": 24,
            "max_temp": 28,
        },
    }

    def predict_zoning(
        self,
        db: Session,
        crop_id: str,
        municipality_id: str,
    ) -> Dict:
        """Predict zoning suitability for a crop in a municipality (mock)."""
        rules = self._SUITABILITY_RULES.get(crop_id)
        if not rules:
            return {
                "crop_id": crop_id,
                "municipality_id": municipality_id,
                "suitability": "low",
                "confidence": 0.5,
                "model_version": "mock-v1",
                "factors": {
                    "temperature_match": False,
                    "precipitation_match": False,
                    "soil_match": False,
                    "altitude_match": False,
                },
            }

        climate = get_municipality_climate_summary(db, municipality_id)
        avg_temp = climate["avg_temp"]

        temp_match = rules["min_temp"] <= avg_temp <= rules["max_temp"]

        min_precipitation = 1.0
        precip_match = climate["has_data"] and climate["avg_precipitation"] >= min_precipitation

        municipality = municipality_catalog.get_municipality_by_id(db, municipality_id)
        municipality_altitude = municipality.altitude if municipality and municipality.altitude is not None else 0
        altitude_match = rules["min_alt"] <= municipality_altitude <= rules["max_alt"]

        soil_match = True

        matches = sum([temp_match, precip_match, altitude_match])
        if matches >= 3:
            suitability = "high"
            confidence = random.uniform(0.85, 0.95)
        elif matches >= 2:
            suitability = "medium"
            confidence = random.uniform(0.70, 0.85)
        else:
            suitability = "low"
            confidence = random.uniform(0.50, 0.70)

        return {
            "crop_id": crop_id,
            "municipality_id": municipality_id,
            "suitability": suitability,
            "confidence": round(confidence, 2),
            "model_version": "mock-v1",
            "factors": {
                "temperature_match": temp_match,
                "precipitation_match": precip_match,
                "soil_match": soil_match,
                "altitude_match": altitude_match,
            },
        }

    def predict_calendar(
        self,
        crop_id: str,
        municipality_id: str,
        month: int,
        year: int,
        planting_months: list[int],
    ) -> Dict:
        """Predict calendar planting ratings for a month (mock)."""
        days = []
        seed = hash(f"{crop_id}-{municipality_id}-{month}-{year}")
        random.seed(seed)

        is_planting_month = month in planting_months

        for day in range(1, 31):
            if is_planting_month:
                rand_val = random.random()
                if rand_val > 0.3:
                    rating = "ideal"
                elif rand_val > 0.1:
                    rating = "acceptable"
                else:
                    rating = "notRecommended"
            else:
                rand_val = random.random()
                if rand_val > 0.7:
                    rating = "acceptable"
                elif rand_val > 0.4:
                    rating = "notRecommended"
                else:
                    rating = "notRecommended"

            days.append({"day": day, "rating": rating})

        ideal_count = sum(1 for d in days if d["rating"] == "ideal")

        return {
            "crop_id": crop_id,
            "municipality_id": municipality_id,
            "month": month,
            "year": year,
            "days": days,
            "ideal_count": ideal_count,
            "model_version": "mock-v1",
        }
