from pydantic import BaseModel, Field
from typing import Optional


class RootResponse(BaseModel):
    """Public metadata exposed at API root."""

    message: str = Field(description="Service name")
    version: str = Field(description="API semantic version")
    docs: str = Field(description="Relative URL to Swagger UI")


class HealthResponse(BaseModel):
    """Basic readiness endpoint used by local tooling and containers."""

    status: str = Field(description="Service health status")
    version: str = Field(description="API semantic version")
    models_loaded: bool = Field(description="Whether production ML models are currently loaded")


class ComponentStatus(BaseModel):
    """Status of a single infrastructure component."""

    name: str
    ready: bool
    detail: Optional[str] = None


class ReadinessResponse(BaseModel):
    """Detailed readiness for all infrastructure components."""

    status: str = Field(description="Overall status: ok or degraded")
    version: str = Field(description="API semantic version")
    components: list[ComponentStatus] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Standard API error payload returned by HTTPException."""

    detail: str = Field(description="Human-readable error message")
