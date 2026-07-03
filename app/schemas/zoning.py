from pydantic import BaseModel


class ZoningRequest(BaseModel):
    crop_id: str
    municipality_id: str


class ZoningFactors(BaseModel):
    temperature_match: bool
    precipitation_match: bool
    soil_match: bool
    altitude_match: bool


class ZoningResponse(BaseModel):
    crop_id: str
    municipality_id: str
    suitability: str  # "high", "medium", "low", "none"
    confidence: float
    model_version: str
    factors: ZoningFactors
