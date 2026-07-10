from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import MunicipalityClimateForecast
from app.schemas.admin import ClimateSyncStatusResponse
from app.services.climate_scheduler import get_last_sync_status

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get(
    "/climate-sync/status",
    response_model=ClimateSyncStatusResponse,
    summary="Get climate sync status",
    description=(
        "Returns the latest climate synchronization run and forecast coverage counters.\n\n"
        "Use cases:\n"
        "- Monitor scheduled sync health from operations dashboards.\n"
        "- Verify data freshness before enabling weather-dependent features."
    ),
)
def climate_sync_status(db: Session = Depends(get_db)):
    last_sync = get_last_sync_status(db)

    forecast_count = db.query(MunicipalityClimateForecast).count()
    distinct_municipalities = (
        db.query(MunicipalityClimateForecast.municipality_id)
        .distinct()
        .count()
    )

    return {
        "last_sync": last_sync,
        "forecast_records": forecast_count,
        "municipalities_with_forecast": distinct_municipalities,
    }
