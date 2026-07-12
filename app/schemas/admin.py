from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List


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
    municipalities_with_forecast: int = Field(
        description="Number of municipalities with at least one forecast record"
    )


class ModelReleaseResponse(BaseModel):
    """Active model release information."""

    id: int
    model_type: str
    crop_key: Optional[str] = None
    model_family: str
    hf_repo: str
    hf_revision: str
    artifact_filename: str
    is_active: bool
    sha256: Optional[str] = None
    preprocessor_version: Optional[str] = None


class ModelStatusResponse(BaseModel):
    """Status of all loaded ML models."""

    models_loaded: bool
    zoning_models: List[ModelReleaseResponse] = Field(default_factory=list)
    yield_models: List[ModelReleaseResponse] = Field(default_factory=list)
    profiles_loaded: bool = Field(default=False, description="Whether reference Parquet profiles are loaded")
    golden_vectors_passed: Optional[bool] = Field(default=None, description="Whether golden vector validation passed")


class CacheStatsResponse(BaseModel):
    """Prediction cache statistics."""

    total_entries: int
    active_entries: int
    expired_entries: int
    by_type: dict[str, int] = Field(default_factory=dict)


class CacheInvalidateRequest(BaseModel):
    """Request to invalidate cache entries."""

    prediction_type: Optional[str] = Field(default=None, description="Invalidate only this prediction type")
    scope_key: Optional[str] = Field(default=None, description="Invalidate only this scope")


class CacheInvalidateResponse(BaseModel):
    """Result of a cache invalidation request."""

    invalidated: int = Field(description="Number of cache entries deleted")


class PredictionRunResponse(BaseModel):
    """Single prediction audit entry."""

    id: int = Field(description="Prediction run identifier")
    request_id: str = Field(description="Caller-facing request id")
    prediction_type: str = Field(description="Prediction type (zoning, yield, calendar)")
    cache_hit: bool = Field(description="Whether the result was served from cache")
    method: str = Field(description="Method used: primary_model, fallback, mock")
    fallback_used: bool = Field(description="Whether a fallback/mock path was used")
    latency_ms: int = Field(description="End-to-end latency in milliseconds")
    status: str = Field(description="Execution status: success, cached or error")
    error_message: Optional[str] = Field(default=None, description="Error details when status is error")
    created_at: Optional[str] = Field(default=None, description="UTC timestamp in ISO 8601 format")
