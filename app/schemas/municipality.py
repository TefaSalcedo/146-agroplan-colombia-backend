from pydantic import BaseModel
from typing import Optional


class MunicipalityResponse(BaseModel):
    id: str
    name: str
    department: str
    lat: float
    lng: float
    altitude: int
    avg_temperature: Optional[float] = None
    precipitation: Optional[float] = None
    dane_code: str

    class Config:
        from_attributes = True


class MunicipalityListResponse(BaseModel):
    municipalities: list[MunicipalityResponse]
    count: int
