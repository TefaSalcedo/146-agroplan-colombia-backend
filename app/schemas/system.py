from pydantic import BaseModel, Field


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


class ErrorResponse(BaseModel):
    """Standard API error payload returned by HTTPException."""

    detail: str = Field(description="Human-readable error message")
