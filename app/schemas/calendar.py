from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class CalendarDay(BaseModel):
    day: int = Field(ge=1, le=31, description="Day of month")
    rating: Literal["ideal", "acceptable", "notRecommended"]


class CalendarRequest(BaseModel):
    crop_id: str = Field(description="Crop identifier")
    municipality_id: str = Field(description="Municipality DANE code (5 digits)")
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


# ---------------------------------------------------------------------------
# Batch calendar endpoint
# ---------------------------------------------------------------------------

class CalendarBatchRequest(BaseModel):
    municipality_id: str = Field(description="Municipality DANE code (5 digits)")
    crop_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of crop IDs. If omitted, all ML-supported crops are evaluated.",
    )
    horizon_months: int = Field(default=12, ge=1, le=12, description="Forecast horizon in months")


class CalendarMonthlyForecast(BaseModel):
    """Climate forecast for a single month in the horizon."""

    month: int = Field(ge=1, le=12)
    year: int = Field(ge=2000, le=2100)
    temp_mean: Optional[float] = None
    precipitation: Optional[float] = None
    humidity: Optional[float] = None
    climate_source: Literal["open_meteo_forecast", "not_available"] = Field(
        description="Source of climate data for this month: open_meteo_forecast or not_available"
    )


class CalendarHarvestWindow(BaseModel):
    """A recommended harvest month with computed planting windows."""

    harvest_month: int = Field(ge=1, le=12)
    harvest_year: int = Field(ge=2000, le=2100)
    harvest_month_name: str
    score: float = Field(ge=0.0, le=1.0, description="Relative ranking score")
    planting_months: List[int] = Field(
        default_factory=list, description="Recommended planting months (may be empty if no data)"
    )
    planting_year: Optional[int] = None
    duration_days_min: Optional[int] = None
    duration_days_max: Optional[int] = None


class CalendarCropResult(BaseModel):
    """Calendar prediction for a single crop in a batch request."""

    crop_id: str
    crop_name: str
    yield_prediction: Optional[float] = Field(
        default=None, description="Predicted yield (t/ha) from ensemble model"
    )
    yield_model_version: Optional[str] = None
    yield_confidence: Optional[str] = Field(
        default=None, description="Confidence level: high, medium, low"
    )
    top_harvest_months: List[CalendarHarvestWindow] = Field(
        default_factory=list, description="Top 3 recommended harvest months"
    )
    monthly_forecasts: List[CalendarMonthlyForecast] = Field(
        default_factory=list, description="Climate data used for each month in the horizon"
    )
    warnings: List[str] = Field(default_factory=list)
    method: str = Field(description="Prediction method used")


class CalendarBatchResponse(BaseModel):
    """Batch calendar response for multiple crops in a municipality."""

    municipality_id: str
    municipality_name: str
    horizon_months: int
    results: List[CalendarCropResult]
    model_version: str
    explanation: Optional[str] = Field(
        default=None, description="LLM-generated explanation (null if LLM unavailable)"
    )
    llm_status: Optional[str] = Field(
        default=None, description="LLM generation status: success, llm_unavailable"
    )
