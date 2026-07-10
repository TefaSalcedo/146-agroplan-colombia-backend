from typing import List
from pydantic import BaseModel, Field

from app.schemas.crop import CropResponseLite, TopCropResponse


class RecommendationRequest(BaseModel):
    municipality_id: str = Field(description="Municipality identifier")


class NextPlantingSeason(BaseModel):
    month: int = Field(ge=1, le=12)
    month_name: str
    crops: List[str]


class RecommendationResponse(BaseModel):
    top_crop: TopCropResponse
    other_crops: List[CropResponseLite]
    next_planting_season: NextPlantingSeason
