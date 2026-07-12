import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from sqlalchemy.orm import Session

from app.models import MunicipalityClimateForecast, Municipality


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def get_municipality_climate_summary(db: Session, municipality_id: str) -> Dict:
    """Compute average climate values from stored forecasts for a municipality."""
    today = _utcnow().date()
    future_cutoff = today + timedelta(days=90)

    records = (
        db.query(MunicipalityClimateForecast)
        .filter(MunicipalityClimateForecast.municipality_dane_code == municipality_id)
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


def _build_climate_map(db: Session, municipality_dane_codes: List[str]) -> Dict[str, Dict]:
    """Fetch climate summaries for many municipalities in a single query."""
    today = _utcnow().date()
    future_cutoff = today + timedelta(days=90)

    records = (
        db.query(MunicipalityClimateForecast)
        .filter(MunicipalityClimateForecast.municipality_dane_code.in_(municipality_dane_codes))
        .filter(MunicipalityClimateForecast.forecast_date >= today)
        .filter(MunicipalityClimateForecast.forecast_date <= future_cutoff)
        .all()
    )

    grouped: Dict[str, List[MunicipalityClimateForecast]] = {}
    for record in records:
        grouped.setdefault(record.municipality_dane_code, []).append(record)

    summaries = {}
    for dane_code in municipality_dane_codes:
        m_records = grouped.get(dane_code, [])
        if not m_records:
            summaries[dane_code] = {"avg_temp": 0.0, "avg_precipitation": 0.0, "has_data": False}
            continue

        temps = [r.temp_mean for r in m_records if r.temp_mean is not None]
        precipitations = [r.precipitation for r in m_records if r.precipitation is not None]

        avg_temp = sum(temps) / max(1, len(temps))
        avg_precipitation = sum(precipitations) / max(1, len(precipitations))

        summaries[dane_code] = {
            "avg_temp": round(avg_temp, 2),
            "avg_precipitation": round(avg_precipitation, 2),
            "has_data": True,
        }

    return summaries


class MockPredictor:
    """Mock predictor for zoning and calendar predictions.

    This provides mock predictions based on simple rules while the real ML models
    are being integrated. When models are available, this will be replaced by
    the prediction service.
    """

    _SUITABILITY_RULES = {
        "aguacate": {
            "min_alt": 1500,
            "max_alt": 2500,
            "min_temp": 16,
            "max_temp": 24,
        },
        "algodon": {
            "min_alt": 0,
            "max_alt": 1000,
            "min_temp": 21,
            "max_temp": 30,
        },
        "cana_panelera": {
            "min_alt": 0,
            "max_alt": 1800,
            "min_temp": 20,
            "max_temp": 28,
        },
        "cebolla": {
            "min_alt": 0,
            "max_alt": 2800,
            "min_temp": 13,
            "max_temp": 25,
        },
        "fresa": {
            "min_alt": 1200,
            "max_alt": 2600,
            "min_temp": 10,
            "max_temp": 22,
        },
        "pina": {
            "min_alt": 0,
            "max_alt": 1200,
            "min_temp": 20,
            "max_temp": 30,
        },
        "soya": {
            "min_alt": 0,
            "max_alt": 1500,
            "min_temp": 20,
            "max_temp": 30,
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

        municipality = db.query(Municipality).filter(
            Municipality.dane_code == municipality_id
        ).first()
        municipality_altitude = (
            municipality.altitude if municipality and municipality.altitude is not None else 0
        )
        altitude_match = rules["min_alt"] <= municipality_altitude <= rules["max_alt"]

        soil_match = True

        matches = sum([temp_match, precip_match, altitude_match])
        if matches >= 3:
            suitability = "high"
            confidence = 0.90
        elif matches >= 2:
            suitability = "medium"
            confidence = 0.75
        else:
            suitability = "low"
            confidence = 0.60

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

    def predict_zoning_batch(
        self,
        db: Session,
        crop_id: str,
        municipalities: List[Municipality],
    ) -> List[Dict]:
        """Predict zoning suitability for a list of municipalities efficiently."""
        rules = self._SUITABILITY_RULES.get(crop_id)
        dane_codes = [m.dane_code for m in municipalities]
        climate_map = _build_climate_map(db, dane_codes) if rules else {}

        predictions = []
        for municipality in municipalities:
            if not rules:
                predictions.append({
                    "crop_id": crop_id,
                    "municipality_id": municipality.dane_code,
                    "suitability": "low",
                    "confidence": 0.5,
                    "model_version": "mock-v1",
                    "factors": {
                        "temperature_match": False,
                        "precipitation_match": False,
                        "soil_match": False,
                        "altitude_match": False,
                    },
                })
                continue

            climate = climate_map.get(
                municipality.dane_code,
                {"avg_temp": 0.0, "avg_precipitation": 0.0, "has_data": False},
            )
            avg_temp = climate["avg_temp"]
            temp_match = rules["min_temp"] <= avg_temp <= rules["max_temp"]

            min_precipitation = 1.0
            precip_match = climate["has_data"] and climate["avg_precipitation"] >= min_precipitation

            municipality_altitude = municipality.altitude if municipality.altitude is not None else 0
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

            predictions.append({
                "crop_id": crop_id,
                "municipality_id": municipality.dane_code,
                "suitability": suitability,
                "confidence": round(confidence, 2),
                "model_version": "mock-v1",
                "factors": {
                    "temperature_match": temp_match,
                    "precipitation_match": precip_match,
                    "soil_match": soil_match,
                    "altitude_match": altitude_match,
                },
            })

        return predictions

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
