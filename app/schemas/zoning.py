from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, List


class ClimateBasedRecommendation(BaseModel):
    """Crop recommendation derived from climate+soil k-NN analogs.

    This is independent of the primary LightGBM model and highlights crops that
    prosper in municipalities with similar climate and soil characteristics.
    """

    crop_id: str = Field(description="Crop identifier")
    crop_name: str = Field(description="Crop common name")
    score: float = Field(ge=0.0, le=1.0, description="Share of apta neighbors that recommend this crop")
    source: str = Field(default="climate_analog_knn", description="Source model: climate_analog_knn")


class ZoningFactors(BaseModel):
    temperature_match: bool = Field(description="Temperature range compatibility")
    precipitation_match: bool = Field(description="Precipitation compatibility")
    soil_match: bool = Field(description="Soil compatibility")
    altitude_match: bool = Field(description="Altitude compatibility")


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
    climate_based_recommendations: List[ClimateBasedRecommendation] = Field(
        default=[],
        description="Additional crops recommended by climate+soil k-NN analogs",
    )
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


class AIGuideAlternativeCrop(BaseModel):
    """Alternative crop suggested by the LLM for a municipality."""

    crop_name: str = Field(description="Crop common name in plain Spanish")
    why: str = Field(description="Why this crop fits the municipality, in farmer-friendly language")
    confidence: str = Field(default="medium", description="high, medium or low")


class AIGuideFarmingSystem(BaseModel):
    """Farming system recommendation for a municipality."""

    title: str = Field(description="System name, e.g. Vivero, Hidroponía")
    recommendation: str = Field(description="Practical recommendation for the farmer")
    suitable: str = Field(default="yes", description="yes, partial or no")


class AIGuideSoilTip(BaseModel):
    """Soil or fertilizer recommendation for a municipality."""

    title: str = Field(description="Tip title")
    content: str = Field(description="Farmer-friendly explanation")


class MunicipalityAIGuideResponse(BaseModel):
    """LLM-generated AI insights for a municipality."""

    municipality_id: str = Field(description="Municipality DANE code")
    municipality_name: str = Field(description="Municipality name")
    summary: str = Field(description="Short friendly summary")
    alternative_crops: List[AIGuideAlternativeCrop] = Field(
        default=[], description="Crops recommended by the AI that are not in the catalog"
    )
    farming_systems: List[AIGuideFarmingSystem] = Field(
        default=[], description="Farming-system recommendations for the municipality"
    )
    soil_and_fertilizer: List[AIGuideSoilTip] = Field(
        default=[], description="Soil and fertilizer tips in plain language"
    )
    generated_at: Optional[str] = Field(default=None, description="ISO timestamp when generated")
    expires_at: Optional[str] = Field(default=None, description="ISO timestamp when expires")
    cached: bool = Field(default=False, description="Whether the response was served from cache")
    provider: Optional[str] = Field(default=None, description="LLM provider used")
    model: Optional[str] = Field(default=None, description="LLM model used")
    tokens_in: Optional[int] = Field(default=None)
    tokens_out: Optional[int] = Field(default=None)
    tokens_total: Optional[int] = Field(default=None)
    latency_ms: Optional[int] = Field(default=None)
    status: str = Field(default="success", description="success or llm_unavailable")
    error: Optional[str] = Field(default=None)



