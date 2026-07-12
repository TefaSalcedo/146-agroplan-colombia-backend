from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class MunicipalityResponse(BaseModel):
    id: str = Field(description="Unique municipality DANE code (5 digits)")
    name: str = Field(description="Municipality display name")
    department: str = Field(description="Department name (backwards compatible)")
    department_id: Optional[str] = Field(default=None, description="Department DANE code (2 digits)")
    lat: float = Field(description="Latitude")
    lng: float = Field(description="Longitude")
    altitude: int = Field(default=0, description="Altitude above sea level in meters")
    avg_temperature: Optional[float] = None
    precipitation: Optional[float] = None
    dane_code: str = Field(description="Official DANE code")
    distance_km: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class MunicipalityListResponse(BaseModel):
    """Response for municipality catalog listings."""

    municipalities: list[MunicipalityResponse]
    count: int


class DepartmentResponse(BaseModel):
    """Department with DANE code and municipality count."""

    dane_code: str = Field(description="Department DANE code (2 digits)")
    name: str = Field(description="Department name")
    municipality_count: int = Field(description="Number of municipalities in this department")


class DepartmentListResponse(BaseModel):
    """List of departments available in the catalog."""

    departments: list[str] = Field(default_factory=list, description="Department names (backwards compatible)")
    departments_detailed: list[DepartmentResponse] = Field(
        default_factory=list, description="Departments with DANE codes and counts"
    )
    count: int = Field(default=0, description="Total number of departments")
