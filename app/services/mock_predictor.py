from typing import Dict
import random
from app.schemas.zoning import ZoningFactors
from app.schemas.calendar import CalendarDay


class MockPredictor:
    """Mock predictor for zoning and calendar predictions.
    
    This provides mock predictions based on simple rules while the real ML models
    are being trained. When models are available, this will be replaced by model_loader.py.
    """
    
    # Mock suitability data based on crop-altitude-temperature compatibility
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
        crop_id: str, 
        municipality_id: str,
        municipality_altitude: int,
        municipality_avg_temp: float
    ) -> Dict:
        """Predict zoning suitability for a crop in a municipality (mock)"""
        
        rules = self._SUITABILITY_RULES.get(crop_id)
        if not rules:
            # Unknown crop - return low suitability
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
                }
            }
        
        # Check altitude match
        alt_match = rules["min_alt"] <= municipality_altitude <= rules["max_alt"]
        
        # Check temperature match
        temp_match = rules["min_temp"] <= municipality_avg_temp <= rules["max_temp"]
        
        # Mock precipitation and soil as always matching for now
        precip_match = True
        soil_match = True
        
        # Calculate suitability based on matches
        if alt_match and temp_match:
            suitability = "high"
            confidence = random.uniform(0.85, 0.95)
        elif alt_match or temp_match:
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
                "altitude_match": alt_match,
            }
        }
    
    def predict_calendar(
        self,
        crop_id: str,
        municipality_id: str,
        month: int,
        year: int,
        planting_months: list[int]
    ) -> Dict:
        """Predict calendar planting ratings for a month (mock)"""
        
        days = []
        seed = hash(f"{crop_id}-{municipality_id}-{month}-{year}")
        random.seed(seed)
        
        # If month is in planting months, more ideal days
        is_planting_month = month in planting_months
        
        for day in range(1, 31):
            if is_planting_month:
                # More ideal days during planting season
                rand_val = random.random()
                if rand_val > 0.3:
                    rating = "ideal"
                elif rand_val > 0.1:
                    rating = "acceptable"
                else:
                    rating = "notRecommended"
            else:
                # Fewer ideal days outside planting season
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
            "model_version": "mock-v1"
        }
