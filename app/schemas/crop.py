from pydantic import BaseModel, Field
from typing import Optional, List


class GrowthStage(BaseModel):
    """Single growth stage description for a crop profile."""

    label: str = Field(description="Stage name")
    icon: str = Field(description="Frontend icon token")
    description: str = Field(description="Detailed stage description")


class Tip(BaseModel):
    """Agronomic tip for a crop."""

    title: str = Field(description="Tip title")
    description: str = Field(description="Tip details")


class CropResponse(BaseModel):
    """Full crop profile returned by the catalog endpoints."""

    id: str = Field(description="Crop identifier")
    name: str = Field(description="Crop common name")
    scientific_name: str = Field(description="Scientific name")
    image: str = Field(description="URL or path to crop image")
    days_to_harvest: int = Field(description="Typical days to harvest")
    soil_type: str = Field(description="Recommended soil type")
    ideal_temperature: str = Field(description="Ideal temperature range")
    humidity: str = Field(description="Ideal humidity range")
    precipitation: str = Field(description="Ideal precipitation range")
    altitude: str = Field(description="Ideal altitude range")
    irrigation: str = Field(description="Irrigation recommendation")
    substrates: List[str] = Field(description="Recommended substrates or growing media")
    planting_months: List[int] = Field(description="Recommended planting months (1-12)")
    harvest_months: List[int] = Field(description="Typical harvest months (1-12)")
    stages: List[GrowthStage] = Field(description="Crop growth stages")
    tips: List[Tip] = Field(description="Agronomic tips")


class CropResponseLite(BaseModel):
    """Lightweight crop entry for lists and cards."""

    id: str = Field(description="Crop identifier")
    name: str = Field(description="Crop common name")
    image: str = Field(description="URL or path to crop image")


class CropListResponse(BaseModel):
    """List of full crop profiles."""

    crops: List[CropResponse] = Field(description="Crop profiles")
    count: int = Field(description="Total number of crops")


class CropRecommendationResponse(BaseModel):
    """LLM-generated recommendation for a specific crop in a municipality."""

    crop_id: str = Field(description="Crop identifier")
    crop_name: str = Field(description="Crop common name")
    municipality_id: str = Field(description="Municipality DANE code")
    municipality_name: str = Field(description="Municipality name")
    text: str = Field(description="Plain-language recommendation for the farmer")
    cached: bool = Field(default=False, description="Whether the response was served from cache")
    generated_at: Optional[str] = Field(default=None, description="ISO timestamp when the recommendation was generated")
    expires_at: Optional[str] = Field(default=None, description="ISO timestamp when the cached recommendation expires")
    provider: Optional[str] = Field(default=None, description="LLM provider used")
    model: Optional[str] = Field(default=None, description="LLM model used")
    tokens_in: Optional[int] = Field(default=None, description="Input tokens consumed")
    tokens_out: Optional[int] = Field(default=None, description="Output tokens consumed")
    tokens_total: Optional[int] = Field(default=None, description="Total tokens consumed")
    latency_ms: Optional[int] = Field(default=None, description="LLM call latency in milliseconds")
    status: str = Field(description="LLM generation status: success or llm_unavailable")
    error: Optional[str] = Field(default=None, description="Error message if generation failed")


class CropNationalGuideSection(BaseModel):
    """Single section of a farmer-friendly national crop guide."""

    title: str = Field(description="Section title")
    content: str = Field(description="Farmer-friendly explanation for this section")


class CropNationalGuideResponse(BaseModel):
    """LLM-generated national farmer guide for a crop, cached for 3 months."""

    crop_id: str = Field(description="Crop identifier")
    crop_name: str = Field(description="Crop common name")
    summary: str = Field(description="Short friendly summary of the guide")
    sections: List[CropNationalGuideSection] = Field(description="Structured guide sections")
    generated_at: Optional[str] = Field(default=None, description="ISO timestamp when the guide was generated")
    expires_at: Optional[str] = Field(default=None, description="ISO timestamp when the guide expires (3 months)")
    cached: bool = Field(default=False, description="Whether the response was served from cache")
    provider: Optional[str] = Field(default=None, description="LLM provider used")
    model: Optional[str] = Field(default=None, description="LLM model used")
    tokens_in: Optional[int] = Field(default=None, description="Input tokens consumed")
    tokens_out: Optional[int] = Field(default=None, description="Output tokens consumed")
    tokens_total: Optional[int] = Field(default=None, description="Total tokens consumed")
    latency_ms: Optional[int] = Field(default=None, description="LLM call latency in milliseconds")
    status: str = Field(default="success", description="LLM generation status: success or llm_unavailable")
    error: Optional[str] = Field(default=None, description="Error message if generation failed")
