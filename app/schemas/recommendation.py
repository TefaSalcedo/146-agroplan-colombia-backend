from pydantic import BaseModel
from typing import List
from app.schemas.crop import CropResponse, CropResponseLite


class RecommendationRequest(BaseModel):
    municipality_id: str


class NextPlantingSeason(BaseModel):
    month: int
    month_name: str
    crops: List[str]


class RecommendationResponse(BaseModel):
    top_crop: CropResponse
    other_crops: List[CropResponseLite]
    next_planting_season: NextPlantingSeason
