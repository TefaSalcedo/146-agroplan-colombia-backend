from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Municipality, MunicipalityClimateForecast
from app.schemas.alerts import AlertResponse
from app.schemas.system import ErrorResponse

router = APIRouter(prefix="/alerts", tags=["alerts"])


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
    municipality = db.query(Municipality).filter(Municipality.dane_code == municipality_id).first()
    if not municipality:
        raise HTTPException(status_code=404, detail="Municipality not found")

    today = datetime.now(timezone.utc).date()
    forecasts = (
        db.query(MunicipalityClimateForecast)
        .filter(MunicipalityClimateForecast.municipality_dane_code == municipality_id)
        .filter(MunicipalityClimateForecast.forecast_date >= today)
        .filter(MunicipalityClimateForecast.forecast_date <= today + timedelta(days=14))
        .all()
    )

    return _generate_alerts_from_forecasts(municipality, forecasts)
