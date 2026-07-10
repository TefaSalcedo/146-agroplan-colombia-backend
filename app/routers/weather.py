from fastapi import APIRouter, Depends, HTTPException, Path, status
from httpx import HTTPError
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.municipality_catalog import MunicipalityCatalog
from app.services.open_meteo import OpenMeteoService
from app.schemas.weather import WeatherResponse
from app.schemas.system import ErrorResponse

router = APIRouter(prefix="/weather", tags=["weather"])
municipality_catalog = MunicipalityCatalog()
weather_service = OpenMeteoService()


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
async def get_weather(
    municipality_id: str = Path(..., description="AgroPlan municipality ID"),
    db: Session = Depends(get_db),
):
    municipality = municipality_catalog.get_municipality_by_id(db, municipality_id)
    if not municipality:
        raise HTTPException(status_code=404, detail="Municipality not found")

    try:
        weather_data = await weather_service.get_current_weather(
            lat=municipality.lat,
            lng=municipality.lng,
        )
    except HTTPError as exc:
        raise HTTPException(status_code=502, detail="Weather provider unavailable") from exc

    return WeatherResponse(**weather_data)
