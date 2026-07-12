from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class AlertResponse(BaseModel):
    id: str = Field(description="Alert identifier")
    level: Literal["info", "warning", "danger"] = Field(description="Severity level")
    title: str = Field(description="Short alert title")
    description: str = Field(description="Detailed recommendation for the farmer")


class ForecastDayResponse(BaseModel):
    date: date
    temp_min: float | None = Field(default=None, description="Minimum temperature (°C)")
    temp_max: float | None = Field(default=None, description="Maximum temperature (°C)")
    temp_mean: float | None = Field(default=None, description="Mean temperature (°C)")
    precipitation: float | None = Field(default=None, description="Daily precipitation (mm)")
    humidity: float | None = Field(default=None, description="Relative humidity (%)")
    uv_index: float | None = Field(default=None, description="Maximum UV index")
    wind_speed: float | None = Field(default=None, description="Maximum wind speed (km/h)")
