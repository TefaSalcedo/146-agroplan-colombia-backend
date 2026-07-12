from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.logger import get_logger
from app.models import Municipality, MunicipalityClimateForecast
from app.schemas.alerts import AlertResponse
from app.schemas.system import ErrorResponse
from app.services.climate_data_service import ClimateDataService

router = APIRouter(prefix="/alerts", tags=["alerts"])
logger = get_logger("app.routers.alerts")
climate_data_service = ClimateDataService()


def _generate_alerts_from_forecasts(
    municipality: Municipality, forecasts: list[MunicipalityClimateForecast]
) -> list[AlertResponse]:
    """Generate simple climate alerts from stored forecast records."""
    alerts: list[AlertResponse] = []
    today = datetime.now(timezone.utc).date()

    if not forecasts:
        alerts.append(
            AlertResponse(
                id=f"{municipality.id}-no-data",
                level="info",
                title="Sin datos de pronóstico",
                description="Aún no hay pronósticos sincronizados para tu municipio. Te mostraremos alertas generales mientras actualizamos la información.",
            )
        )
        return alerts

    next_7_days = [f for f in forecasts if today <= f.forecast_date <= today + timedelta(days=7)]

    high_rain_days = [
        f for f in next_7_days
        if f.precipitation is not None and f.precipitation >= 15.0
    ]
    hot_days = [
        f for f in next_7_days
        if f.temp_max is not None and f.temp_max >= 30.0
    ]
    dry_days = [
        f for f in next_7_days
        if f.precipitation is not None and f.precipitation == 0.0
    ]

    if high_rain_days:
        alerts.append(
            AlertResponse(
                id=f"{municipality.id}-heavy-rain",
                level="warning",
                title="Lluvias fuertes próximas",
                description=f"Se esperan {len(high_rain_days)} días con lluvias intensas (>15 mm) en la próxima semana. Revisa el drenaje de tu cultivo.",
            )
        )

    if hot_days:
        alerts.append(
            AlertResponse(
                id=f"{municipality.id}-heat",
                level="warning",
                title="Olas de calor",
                description=f"{len(hot_days)} días superarán los 30 °C. Mantén la humedad del suelo y considera sombra para cultivos sensibles.",
            )
        )

    if len(dry_days) >= 3:
        alerts.append(
            AlertResponse(
                id=f"{municipality.id}-dry-spell",
                level="danger",
                title="Periodo seco",
                description=f"Se prevén {len(dry_days)} días sin lluvia seguidos. Planifica el riego para evitar estrés hídrico.",
            )
        )

    if not alerts:
        alerts.append(
            AlertResponse(
                id=f"{municipality.id}-stable",
                level="info",
                title="Condiciones estables",
                description="El pronóstico no muestra eventos extremos en los próximos días. Es un buen momento para revisar tu calendario de labores.",
            )
        )

    return alerts


@router.get(
    "/{municipality_id}",
    response_model=list[AlertResponse],
    summary="Get climate alerts",
    description=(
        "Generates short-term alert cards from stored municipality forecasts.\n\n"
        "Use cases:\n"
        "- Highlight near-term climate risks in dashboards.\n"
        "- Trigger proactive agronomic actions (drainage, irrigation, heat mitigation)."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Municipality ID does not exist.",
        }
    },
)
def get_alerts(
    municipality_id: str = Path(..., description="AgroPlan municipality ID"),
    db: Session = Depends(get_db),
):
    logger.info("[endpoint] GET /alerts/{municipality_id} called (municipality_id=%s)", municipality_id)
    logger.debug("[endpoint] Querying database for municipality_id=%s", municipality_id)
    municipality = db.query(Municipality).filter(Municipality.dane_code == municipality_id).first()
    if not municipality:
        logger.warning("[endpoint] Municipality not found: %s", municipality_id)
        raise HTTPException(status_code=404, detail="Municipality not found")

    logger.debug("[endpoint] Fetching forecast records via ClimateDataService")
    forecasts = climate_data_service.get_forecast_records(
        db=db,
        municipality=municipality,
        days=14,
    )
    logger.debug("[endpoint] Found %s forecast records", len(forecasts))

    alerts = _generate_alerts_from_forecasts(municipality, forecasts)
    logger.info("[endpoint] GET /alerts/{municipality_id} returning %s alerts", len(alerts))
    return alerts
