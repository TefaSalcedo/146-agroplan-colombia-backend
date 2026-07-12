from typing import List
from pydantic import BaseModel, Field

from app.schemas.crop import CropResponseLite, TopCropResponse


class RecommendationRequest(BaseModel):
    municipality_id: str = Field(description="Municipality DANE code (5 digits)")


class NextPlantingSeason(BaseModel):
    month: int = Field(ge=1, le=12, description="Month of the next planting season")
    month_name: str = Field(description="Spanish name of the month")
    crops: List[str] = Field(description="Crop IDs recommended for planting in that month")


class RecommendationResponse(BaseModel):
    top_crop: TopCropResponse = Field(description="Highest ranked crop for the municipality")
    other_crops: List[CropResponseLite] = Field(description="Alternative crops ranked 2-5")
    next_planting_season: NextPlantingSeason = Field(
        description="Crops that can be planted in the upcoming month"
    )
