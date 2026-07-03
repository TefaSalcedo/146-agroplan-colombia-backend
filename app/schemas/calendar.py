from pydantic import BaseModel
from typing import List


class CalendarDay(BaseModel):
    day: int
    rating: str  # "ideal", "acceptable", "notRecommended"


class CalendarRequest(BaseModel):
    crop_id: str
    municipality_id: str
    month: int
    year: int


class CalendarResponse(BaseModel):
    crop_id: str
    municipality_id: str
    month: int
    year: int
    days: List[CalendarDay]
    ideal_count: int
    model_version: str
