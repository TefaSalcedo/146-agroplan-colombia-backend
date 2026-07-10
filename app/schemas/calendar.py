from typing import List, Literal
from pydantic import BaseModel, Field


class CalendarDay(BaseModel):
    day: int = Field(ge=1, le=31, description="Day of month")
    rating: Literal["ideal", "acceptable", "notRecommended"]


class CalendarRequest(BaseModel):
    crop_id: str = Field(description="Crop identifier")
    municipality_id: str = Field(description="Municipality identifier")
    month: int = Field(ge=1, le=12, description="Target month")
    year: int = Field(ge=2000, le=2100, description="Target year")


class CalendarResponse(BaseModel):
    crop_id: str
    municipality_id: str
    month: int
    year: int
    days: List[CalendarDay]
    ideal_count: int
    model_version: str
