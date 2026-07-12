from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.logger import get_logger
from app.models import Municipality, MunicipalityClimateForecast
from app.schemas.alerts import ForecastDayResponse
from app.schemas.system import ErrorResponse
from app.services.climate_data_service import ClimateDataService

from datetime import datetime, timedelta, timezone

router = APIRouter(prefix="/forecast", tags=["forecast"])
logger = get_logger("app.routers.forecast")
climate_data_service = ClimateDataService()


@router.get(
    "/daily/{municipality_id}",
    response_model=list[ForecastDayResponse],
    summary="Get daily forecast",
    description=(
        "Returns stored daily climate forecast values for a municipality.\n\n"
        "Use cases:\n"
        "- Build short-term forecast charts.\n"
        "- Feed rule-based alerting or planning logic."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Municipality ID does not exist.",
        }
    },
)
def get_daily_forecast(
    municipality_id: str = Path(..., description="AgroPlan municipality ID"),
    days: int = Query(7, ge=1, le=90, description="Number of days to return (1-90)"),
    db: Session = Depends(get_db),
):
    logger.info("[endpoint] GET /forecast/daily/{municipality_id} called (municipality_id=%s, days=%s)", municipality_id, days)
    logger.debug("[endpoint] Querying database for municipality_id=%s", municipality_id)
    municipality = db.query(Municipality).filter(Municipality.dane_code == municipality_id).first()
    if not municipality:
        logger.warning("[endpoint] Municipality not found: %s", municipality_id)
        raise HTTPException(status_code=404, detail="Municipality not found")

    records = climate_data_service.get_forecast_records(db=db, municipality=municipality, days=days)
    logger.info("[endpoint] GET /forecast/daily/{municipality_id} returning %s forecast records", len(records))

    return [
        ForecastDayResponse(
            date=record.forecast_date,
            temp_min=record.temp_min,
            temp_max=record.temp_max,
            temp_mean=record.temp_mean,
            precipitation=record.precipitation,
            humidity=record.humidity,
            uv_index=record.uv_index,
            wind_speed=record.wind_speed,
        )
        for record in records
    ]
