from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.models import Municipality, MunicipalityClimateForecast, ClimateSyncLog
from app.routers import municipalities, weather, crops, zoning, calendars, recommendations, admin, alerts, forecast
from app.schemas.system import RootResponse, HealthResponse
from app.services.climate_scheduler import get_scheduler, schedule_climate_jobs

settings = get_settings()

OPENAPI_TAGS = [
    {"name": "system", "description": "Operational endpoints for service metadata and health checks."},
    {"name": "municipalities", "description": "Municipality catalog and geospatial lookup endpoints."},
    {"name": "weather", "description": "Current weather conditions by municipality."},
    {"name": "forecast", "description": "Stored daily forecast time-series from Open-Meteo sync jobs."},
    {"name": "alerts", "description": "Climate risk alerts derived from stored forecast records."},
    {"name": "crops", "description": "Crop catalog endpoints used by recommendation and prediction flows."},
    {"name": "zoning", "description": "Crop suitability prediction (mock model) by municipality."},
    {"name": "calendars", "description": "Planting day ratings for a crop-month combination (mock model)."},
    {"name": "recommendations", "description": "Top crop recommendation ranking for a municipality."},
    {"name": "admin", "description": "Operational status for background climate synchronization."},
]

app = FastAPI(
    title="AgroPlan Colombia Backend",
    description=(
        "AgroPlan Colombia API for municipality intelligence, weather context, "
        "crop recommendations and mock agronomic predictions.\n\n"
        "Use this API to:\n"
        "- Browse municipalities and departments.\n"
        "- Retrieve weather, forecast and climate alerts.\n"
        "- Evaluate zoning suitability and planting calendars.\n"
        "- Rank crop recommendations for decision support."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=OPENAPI_TAGS,
)


@app.on_event("startup")
async def startup_event():
    """Start background climate sync scheduler on application startup."""
    scheduler = get_scheduler()
    schedule_climate_jobs(scheduler)
    scheduler.start()


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
    description=(
        "Returns service metadata and the relative Swagger URL.\n\n"
        "Use cases:\n"
        "- Verify the service is reachable.\n"
        "- Discover docs route programmatically."
    ),
)
async def root():
    return {
        "message": "AgroPlan Colombia Backend API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get(
    f"{settings.api_v1_prefix}/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Health check",
    description=(
        "Readiness endpoint for containers, local scripts and uptime checks.\n\n"
        "Use cases:\n"
        "- Kubernetes or Docker health probes.\n"
        "- CI sanity checks after deployment."
    ),
)
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "models_loaded": False,
    }


app.include_router(municipalities.router, prefix=settings.api_v1_prefix)
app.include_router(weather.router, prefix=settings.api_v1_prefix)
app.include_router(crops.router, prefix=settings.api_v1_prefix)
app.include_router(zoning.router, prefix=settings.api_v1_prefix)
app.include_router(calendars.router, prefix=settings.api_v1_prefix)
app.include_router(recommendations.router, prefix=settings.api_v1_prefix)
app.include_router(alerts.router, prefix=settings.api_v1_prefix)
app.include_router(forecast.router, prefix=settings.api_v1_prefix)
app.include_router(admin.router, prefix=settings.api_v1_prefix)
