from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.logger import get_logger
from app.models import Municipality, MunicipalityClimateForecast
from app.schemas.alerts import ForecastDayResponse
from app.schemas.forecast import MonthlyForecastResponse, MonthlyForecastItem
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


@router.get(
    "/monthly/{municipality_id}",
    response_model=MonthlyForecastResponse,
    summary="Get monthly seasonal forecast",
    description=(
        "Returns a monthly seasonal forecast for a municipality using Open-Meteo SEAS5. "
        "Each month includes mean temperature, precipitation and anomalies compared to the "
        "climate average. Useful for telling farmers whether upcoming months are likely to be "
        "warmer, cooler, wetter or drier than normal."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Municipality ID does not exist.",
        }
    },
)
def get_monthly_forecast(
    municipality_id: str = Path(..., description="AgroPlan municipality ID"),
    months: int = Query(4, ge=1, le=7, description="Number of months to forecast (1-7)"),
    db: Session = Depends(get_db),
):
    logger.info("[endpoint] GET /forecast/monthly/{municipality_id} called (municipality_id=%s, months=%s)", municipality_id, months)
    logger.debug("[endpoint] Querying database for municipality_id=%s", municipality_id)
    municipality = db.query(Municipality).filter(Municipality.dane_code == municipality_id).first()
    if not municipality:
        logger.warning("[endpoint] Municipality not found: %s", municipality_id)
        raise HTTPException(status_code=404, detail="Municipality not found")

    records = climate_data_service.get_monthly_forecast_records(
        db=db,
        municipality=municipality,
        months=months,
    )

    spanish_months = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
    ]

    forecasts = []
    for record in records:
        forecast_month = record.forecast_month
        if not forecast_month:
            continue

        forecasts.append(
            MonthlyForecastItem(
                forecast_month=forecast_month,
                month_name=spanish_months[forecast_month.month - 1],
                temp_mean=record.temp_mean,
                temp_anomaly=record.temp_anomaly,
                precipitation=record.precipitation,
                precipitation_anomaly=record.precipitation_anomaly,
                trend=record.trend or "neutral",
                source=record.source,
                fetched_at=record.fetched_at.isoformat() if record.fetched_at else None,
            )
        )

    logger.info("[endpoint] GET /forecast/monthly/{municipality_id} returning %s monthly records", len(forecasts))
    return MonthlyForecastResponse(
        municipality_id=municipality_id,
        municipality_name=municipality.name,
        months=months,
        model="ecmwf_seas5",
        forecasts=forecasts,
    )
