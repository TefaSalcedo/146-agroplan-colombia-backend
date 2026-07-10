from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Municipality, MunicipalityClimateForecast
from app.schemas.alerts import ForecastDayResponse
from app.schemas.system import ErrorResponse

router = APIRouter(prefix="/forecast", tags=["forecast"])


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
    municipality = db.query(Municipality).filter(Municipality.id == municipality_id).first()
    if not municipality:
        raise HTTPException(status_code=404, detail="Municipality not found")

    today = datetime.utcnow().date()
    end_date = today + timedelta(days=days)

    records = (
        db.query(MunicipalityClimateForecast)
        .filter(MunicipalityClimateForecast.municipality_id == municipality_id)
        .filter(MunicipalityClimateForecast.forecast_date >= today)
        .filter(MunicipalityClimateForecast.forecast_date <= end_date)
        .order_by(MunicipalityClimateForecast.forecast_date.asc())
        .all()
    )

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
