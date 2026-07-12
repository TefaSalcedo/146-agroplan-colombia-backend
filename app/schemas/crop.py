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
    success_rate: int = Field(description="Historical success rate (0-100)")
    recommendation: str = Field(description="Short recommendation summary")
    short_reason: str = Field(description="One-line reason for the recommendation")
    reason: str = Field(description="Longer explanation of why the crop is recommended")
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
    recommendation: str = Field(description="Short recommendation summary")
    success_rate: int = Field(description="Historical success rate (0-100)")


class CropListResponse(BaseModel):
    """List of full crop profiles."""

    crops: List[CropResponse] = Field(description="Crop profiles")
    count: int = Field(description="Total number of crops")


class TopCropResponse(CropResponse):
    """Highest ranked crop in a recommendation, including suitability."""

    suitability: str = Field(description="Suitability level: high, medium, low, none")
