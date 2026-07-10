from pydantic import BaseModel, Field
from typing import Literal


class ZoningRequest(BaseModel):
    crop_id: str = Field(description="Crop identifier")
    municipality_id: str = Field(description="Municipality identifier")


class ZoningFactors(BaseModel):
    temperature_match: bool = Field(description="Temperature range compatibility")
    precipitation_match: bool = Field(description="Precipitation compatibility")
    soil_match: bool = Field(description="Soil compatibility")
    altitude_match: bool = Field(description="Altitude compatibility")


class ZoningResponse(BaseModel):
    crop_id: str
    municipality_id: str
    suitability: Literal["high", "medium", "low", "none"]
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str
    factors: ZoningFactors
