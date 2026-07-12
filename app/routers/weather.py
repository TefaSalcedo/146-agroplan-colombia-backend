from fastapi import APIRouter, Depends, HTTPException, Path, status
from httpx import HTTPError
from sqlalchemy.orm import Session
from app.database import get_db
from app.logger import get_logger
from app.services.climate_data_service import ClimateDataService
from app.services.municipality_catalog import MunicipalityCatalog
from app.schemas.weather import WeatherResponse
from app.schemas.system import ErrorResponse

router = APIRouter(prefix="/weather", tags=["weather"])
municipality_catalog = MunicipalityCatalog()
climate_data_service = ClimateDataService()
logger = get_logger("app.routers.weather")


@router.get(
    "/{municipality_id}",
    response_model=WeatherResponse,
    summary="Get current weather",
    description=(
        "Returns near-real-time weather conditions for a municipality using Open-Meteo.\n\n"
        "Use cases:\n"
        "- Show current climate context before recommendations.\n"
        "- Provide weather cards on municipality dashboards."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Municipality ID does not exist.",
        },
        status.HTTP_502_BAD_GATEWAY: {
            "model": ErrorResponse,
            "description": "Weather upstream provider failed or is unavailable.",
        },
    },
)
def get_weather(
    municipality_id: str = Path(..., description="AgroPlan municipality ID"),
    db: Session = Depends(get_db),
):
    logger.info("[endpoint] GET /weather/{municipality_id} called (municipality_id=%s)", municipality_id)
    logger.debug("[endpoint] Querying database for municipality_id=%s", municipality_id)
    municipality = municipality_catalog.get_municipality_by_id(db, municipality_id)
    if not municipality:
        logger.warning("[endpoint] Municipality not found: %s", municipality_id)
        raise HTTPException(status_code=404, detail="Municipality not found")

    logger.info("[endpoint] Calling ClimateDataService for municipality_id=%s", municipality_id)
    try:
        weather_data = climate_data_service.get_current_weather(
            db=db,
            municipality=municipality,
        )
    except HTTPError as exc:
        logger.error("[endpoint] Open-Meteo request failed for municipality_id=%s: %s", municipality_id, exc)
        raise HTTPException(status_code=502, detail="Weather provider unavailable") from exc

    logger.info("[endpoint] GET /weather/{municipality_id} returning weather for municipality_id=%s", municipality_id)
    return WeatherResponse(**weather_data)
