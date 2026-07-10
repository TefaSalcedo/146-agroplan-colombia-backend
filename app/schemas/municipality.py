from pydantic import BaseModel, Field
from typing import Optional


class MunicipalityResponse(BaseModel):
    id: str = Field(description="Unique municipality identifier")
    name: str = Field(description="Municipality display name")
    department: str = Field(description="Department name")
    lat: float = Field(description="Latitude")
    lng: float = Field(description="Longitude")
    altitude: int = Field(description="Altitude above sea level in meters")
    avg_temperature: Optional[float] = None
    precipitation: Optional[float] = None
    dane_code: str = Field(description="Official DANE code")
    distance_km: Optional[float] = None

    class Config:
        from_attributes = True


class MunicipalityListResponse(BaseModel):
    """Paginated-like response for municipality catalog listings."""

    municipalities: list[MunicipalityResponse]
    count: int


class DepartmentListResponse(BaseModel):
    """List of unique departments available in the municipalities catalog."""

    departments: list[str]
