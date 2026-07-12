from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field


class MonthlyForecastItem(BaseModel):
    """Single monthly seasonal forecast record."""

    forecast_month: date = Field(description="First day of the forecast month")
    month_name: str = Field(description="Human-readable month name")
    temp_mean: Optional[float] = Field(default=None, description="Mean temperature (°C)")
    temp_anomaly: Optional[float] = Field(default=None, description="Temperature anomaly vs climate average (°C)")
    precipitation: Optional[float] = Field(default=None, description="Mean monthly precipitation (mm)")
    precipitation_anomaly: Optional[float] = Field(default=None, description="Precipitation anomaly vs climate average (mm)")
    trend: str = Field(default="neutral", description="Qualitative trend: warmer, cooler, wetter, drier, neutral")
    source: str = Field(default="open-meteo-seasonal")
    fetched_at: Optional[str] = Field(default=None, description="ISO timestamp when data was fetched")


class MonthlyForecastResponse(BaseModel):
    """Monthly seasonal forecast response for a municipality."""

    municipality_id: str
    municipality_name: str
    months: int = Field(description="Number of months requested")
    model: str = Field(default="ecmwf_seas5", description="Seasonal forecast model used")
    forecasts: List[MonthlyForecastItem]
    note: str = Field(
        default=(
            "Seasonal forecasts are area-averaged outlooks indicating whether months "
            "are likely to be warmer/cooler or wetter/drier than average. "
            "They are not bias-corrected and should not be interpreted as exact daily predictions."
        ),
        description="Usage note for consumers",
    )
