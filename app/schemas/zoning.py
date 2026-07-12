from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, List


class ZoningRequest(BaseModel):
    crop_id: str = Field(description="Crop identifier")
    municipality_id: str = Field(description="Municipality DANE code (5 digits)")


class ZoningBatchRequest(BaseModel):
    municipality_id: str = Field(description="Municipality DANE code (5 digits)")
    crop_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of crop IDs. If omitted, all ML-supported crops are evaluated.",
    )


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
    # Extended fields (optional for backwards compatibility)
    method: Optional[str] = Field(default=None, description="Prediction method: primary_model, fallback, unavailable")
    probabilities: Optional[Dict[str, float]] = Field(
        default=None, description="Per-class probabilities when available"
    )
    warnings: Optional[List[str]] = Field(default=None, description="Data quality warnings")
    cache_hit: Optional[bool] = Field(default=None, description="Whether this result came from cache")


class ZoningBatchCropResult(BaseModel):
    """Zoning result for a single crop in a batch request."""

    crop_id: str
    crop_name: str
    suitability: Literal["high", "medium", "low", "none"]
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str
    method: str = Field(description="Prediction method: primary_model, fallback, unavailable")
    factors: ZoningFactors
    probabilities: Optional[Dict[str, float]] = None
    warnings: Optional[List[str]] = None


class ZoningBatchResponse(BaseModel):
    """Batch zoning response for all crops in a municipality."""

    municipality_id: str
    municipality_name: str
    results: List[ZoningBatchCropResult]
    model_version: str


class ZoningMapMunicipalityResult(BaseModel):
    """Single municipality result in a zoning map."""

    municipality_id: str
    municipality_name: str
    dane_code: str
    lat: float
    lng: float
    suitability: Literal["high", "medium", "low", "none"]
    confidence: float = Field(ge=0.0, le=1.0)
    method: str
    probabilities: Optional[Dict[str, float]] = None


class ZoningMapResponse(BaseModel):
    """Zoning map for a crop across all municipalities."""

    crop_id: str
    crop_name: str
    model_version: str
    method: str = Field(description="Primary method used across municipalities")
    results: List[ZoningMapMunicipalityResult]
    total_municipalities: int
    cache_hit: Optional[bool] = None
