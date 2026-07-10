from datetime import datetime
from pydantic import BaseModel, Field


class ClimateSyncLogResponse(BaseModel):
    """Execution metadata for a climate sync run."""

    id: int = Field(description="Sync log identifier")
    sync_type: str = Field(description="Sync mode (for example: initial, daily, weekly_extension)")
    status: str = Field(description="Execution status: running, success, partial or error")
    started_at: datetime | None = Field(default=None, description="UTC start timestamp")
    finished_at: datetime | None = Field(default=None, description="UTC finish timestamp")
    records_processed: int = Field(description="Total forecast records processed")
    records_failed: int = Field(description="Total municipalities with failed fetches")
    error_message: str | None = Field(default=None, description="Error details when status is error")


class ClimateSyncStatusResponse(BaseModel):
    """Current climate sync status and forecast coverage counters."""

    last_sync: ClimateSyncLogResponse | None = Field(default=None)
    forecast_records: int = Field(description="Total rows in municipality_climate_forecasts")
    municipalities_with_forecast: int = Field(description="Number of municipalities with at least one forecast record")
