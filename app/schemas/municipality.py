from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional


class MunicipalityResponse(BaseModel):
    """Single municipality with location and climate metadata."""

    id: str = Field(description="Unique municipality DANE code (5 digits)")
    name: str = Field(description="Municipality display name")
    department: str = Field(description="Department name (backwards compatible)")
    department_id: Optional[str] = Field(default=None, description="Department DANE code (2 digits)")
    lat: float = Field(description="Latitude in decimal degrees")
    lng: float = Field(description="Longitude in decimal degrees")
    altitude: int = Field(default=0, description="Altitude above sea level in meters")
    avg_temperature: Optional[float] = Field(default=None, description="Average annual temperature (°C)")
    precipitation: Optional[float] = Field(default=None, description="Average annual precipitation (mm)")
    dane_code: str = Field(description="Official DANE code")
    distance_km: Optional[float] = Field(default=None, description="Distance from provided coordinates when searching nearby")

    model_config = ConfigDict(from_attributes=True)


class MunicipalityListResponse(BaseModel):
    """Response for municipality catalog listings."""

    municipalities: list[MunicipalityResponse] = Field(description="Municipality entries")
    count: int = Field(description="Total number of municipalities returned")


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


class MunicipalitySearchResult(BaseModel):
    """Single result for municipality/department autocomplete."""

    id: str = Field(description="DANE code (municipality 5 digits or department 2 digits)")
    name: str = Field(description="Display name")
    type: Literal["municipality", "department"] = Field(description="Entity type")
    department_id: Optional[str] = Field(default=None, description="Department DANE code when type is municipality")
    department_name: Optional[str] = Field(default=None, description="Department name when type is municipality")


class MunicipalitySearchResponse(BaseModel):
    """Response for municipality/department autocomplete search."""

    query: str = Field(description="Search query")
    results: list[MunicipalitySearchResult]
    count: int
