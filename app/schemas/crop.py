from pydantic import BaseModel
from typing import Optional, List


class GrowthStage(BaseModel):
    label: str
    icon: str
    description: str


class Tip(BaseModel):
    title: str
    description: str


class CropResponse(BaseModel):
    id: str
    name: str
    scientific_name: str
    image: str
    success_rate: int
    recommendation: str
    short_reason: str
    reason: str
    days_to_harvest: int
    soil_type: str
    ideal_temperature: str
    humidity: str
    precipitation: str
    altitude: str
    irrigation: str
    substrates: List[str]
    planting_months: List[int]
    harvest_months: List[int]
    stages: List[GrowthStage]
    tips: List[Tip]


class CropResponseLite(BaseModel):
    id: str
    name: str
    image: str
    recommendation: str
    success_rate: int


class CropListResponse(BaseModel):
    crops: List[CropResponse]
    count: int


class TopCropResponse(CropResponse):
    suitability: str
