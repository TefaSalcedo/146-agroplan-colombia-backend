from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.middleware.rate_limit import GeneralRateLimitMiddleware, RequestBodySizeLimitMiddleware
from app.logger import configure_logging, get_logger
from app.routers import (
    municipalities,
    weather,
    crops,
    zoning,
    calendars,
    admin,
    alerts,
    forecast,
)
from app.schemas.system import (
    RootResponse,
    HealthResponse,
    ReadinessResponse,
    ComponentStatus,
)
from app.services.climate_scheduler import get_scheduler, schedule_climate_jobs
from app.services.model_loader import get_model_loader

# Configure logging as early as possible so every module inherits the level/format.
configure_logging()

logger = get_logger("app.main")
settings = get_settings()

OPENAPI_TAGS = [
    {"name": "system", "description": "Operational endpoints for service metadata and health checks."},
    {"name": "municipalities", "description": "Municipality catalog and geospatial lookup endpoints."},
    {"name": "weather", "description": "Current weather conditions by municipality."},
    {"name": "forecast", "description": "Stored daily forecast time-series from Open-Meteo sync jobs."},
    {"name": "alerts", "description": "Climate risk alerts derived from stored forecast records."},
    {"name": "crops", "description": "Crop catalog endpoints (7 ML-supported crops)."},
    {"name": "zoning", "description": "Crop suitability prediction by municipality with ML models and fallbacks."},
    {"name": "calendars", "description": "Planting calendar predictions (batch multi-crop, 12-month horizon)."},
    {"name": "admin", "description": "Operational status, model health, cache management and audit (API key protected)."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models and start scheduler on startup."""
    logger.info("[startup] Beginning application startup")

    # Load ML models
    logger.info("[startup] Loading ML models and reference artifacts")
    loader = get_model_loader()
    status = loader.get_status()
    if status["zoning_model_loaded"]:
        logger.info("[startup] ML models loaded successfully")
    else:
        logger.error(f"[startup] ML models not loaded: {status.get('load_error', 'no models configured')}")

    # Start climate sync scheduler
    logger.info("[startup] Configuring climate sync scheduler (enable_climate_sync=%s)", settings.enable_climate_sync)
    scheduler = get_scheduler()
    schedule_climate_jobs(scheduler)
    scheduler.start()
    logger.info("[startup] Application startup complete")

    yield

    # Shutdown: stop scheduler
    logger.info("[shutdown] Stopping climate sync scheduler")
    scheduler.shutdown(wait=False)


app = FastAPI(
    title="AgroPlan Colombia Backend",
    description=(
        "AgroPlan Colombia API for municipality intelligence, weather context, "
        "crop zoning and ML-powered agronomic predictions.\n\n"
        "Use this API to:\n"
        "- Browse municipalities and departments.\n"
        "- Retrieve weather, forecast and climate alerts.\n"
        "- Evaluate zoning suitability and planting calendars.\n\n"
        "Supported crops: aguacate, algodon, cana panelera, cebolla, fresa, pina, soya."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
)

app.add_middleware(GeneralRateLimitMiddleware)
app.add_middleware(RequestBodySizeLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    response_model=RootResponse,
    tags=["system"],
    summary="API metadata",
    description="Returns service metadata and the relative Swagger URL.",
)
async def root():
    return {
        "message": "AgroPlan Colombia Backend API",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get(
    f"{settings.api_v1_prefix}/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Health check",
    description="Basic readiness endpoint for containers and uptime checks.",
)
async def health():
    loader = get_model_loader()
    return {
        "status": "ok",
        "version": "2.0.0",
        "models_loaded": loader.is_zoning_model_loaded(),
    }


@app.get(
    f"{settings.api_v1_prefix}/readiness",
    response_model=ReadinessResponse,
    tags=["system"],
    summary="Detailed readiness check",
    description=(
        "Returns the status of all infrastructure components: "
        "database, ML models, reference profiles, and golden vector validation."
    ),
)
async def readiness():
    loader = get_model_loader()
    components = []

    # Check database
    try:
        from app.database import engine
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        components.append(ComponentStatus(name="database", ready=True))
    except Exception:
        components.append(ComponentStatus(name="database", ready=False, detail="Database unavailable"))

    # Check ML models
    components.append(
        ComponentStatus(
            name="zoning_model",
            ready=loader.is_zoning_model_loaded(),
            detail=None if loader.is_zoning_model_loaded() else "Model not loaded",
        )
    )

    components.append(
        ComponentStatus(
            name="yield_models",
            ready=loader.is_yield_model_loaded(),
            detail=None if loader.is_yield_model_loaded() else "Yield models not loaded",
        )
    )

    components.append(
        ComponentStatus(
            name="reference_profiles",
            ready=loader.is_profiles_loaded(),
            detail=None if loader.is_profiles_loaded() else "Profiles not loaded",
        )
    )

    components.append(
        ComponentStatus(
            name="golden_vectors",
            ready=loader.is_golden_vectors_passed() is True,
            detail=None if loader.is_golden_vectors_passed() is True else "Not validated",
        )
    )

    all_ready = all(c.ready for c in components)

    return ReadinessResponse(
        status="ok" if all_ready else "degraded",
        version="2.0.0",
        components=components,
    )


app.include_router(municipalities.router, prefix=settings.api_v1_prefix)
app.include_router(weather.router, prefix=settings.api_v1_prefix)
app.include_router(crops.router, prefix=settings.api_v1_prefix)
app.include_router(zoning.router, prefix=settings.api_v1_prefix)
app.include_router(calendars.router, prefix=settings.api_v1_prefix)
app.include_router(alerts.router, prefix=settings.api_v1_prefix)
app.include_router(forecast.router, prefix=settings.api_v1_prefix)
app.include_router(admin.router, prefix=settings.api_v1_prefix)
