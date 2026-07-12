"""Read-through climate data service.

Endpoints that need Open-Meteo data first check the local database. When data
is missing or stale, the service fetches from Open-Meteo, persists it, and
returns the fresh value. This makes every request act as an incremental sync.
"""

from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.logger import get_logger
from app.models import Municipality, MunicipalityClimateForecast, MunicipalityCurrentWeather
from app.services.open_meteo import OpenMeteoService

logger = get_logger("app.services.climate_data_service")


class ClimateDataService:
    """Read-through cache for municipality climate data."""

    def __init__(self, open_meteo_service: Optional[OpenMeteoService] = None):
        self.open_meteo = open_meteo_service or OpenMeteoService()

    def _today(self) -> date:
        return datetime.now(timezone.utc).date()

    def get_forecast_records(
        self,
        db: Session,
        municipality: Municipality,
        days: int,
    ) -> List[MunicipalityClimateForecast]:
        """Return forecast records for the next ``days`` days, fetching missing days from Open-Meteo.

        The method is forgiving: it returns whatever is already stored plus any
        newly fetched records. If Open-Meteo fails, it returns the stored data
        without raising so that endpoints remain available.
        """
        today = self._today()
        end_date = today + timedelta(days=days)

        logger.debug(
            "[get_forecast_records] Looking up stored forecasts for %s from %s to %s",
            municipality.dane_code,
            today,
            end_date,
        )
        stored = (
            db.query(MunicipalityClimateForecast)
            .filter(MunicipalityClimateForecast.municipality_dane_code == municipality.dane_code)
            .filter(MunicipalityClimateForecast.forecast_date >= today)
            .filter(MunicipalityClimateForecast.forecast_date <= end_date)
            .order_by(MunicipalityClimateForecast.forecast_date.asc())
            .all()
        )
        stored_dates = {record.forecast_date for record in stored}
        requested_dates = {today + timedelta(days=i) for i in range(days)}
        missing_dates = sorted(requested_dates - stored_dates)

        logger.debug(
            "[get_forecast_records] Stored %s records, missing %s dates",
            len(stored),
            len(missing_dates),
        )

        if missing_dates:
            logger.info(
                "[get_forecast_records] Fetching %s missing forecast days for %s from Open-Meteo",
                len(missing_dates),
                municipality.dane_code,
            )
            try:
                fetched = self.open_meteo.get_daily_forecast(
                    lat=municipality.lat,
                    lng=municipality.lng,
                    days=days,
                )
                if fetched:
                    self._upsert_forecast_records(db, municipality.dane_code, fetched)
                    logger.info(
                        "[get_forecast_records] Saved %s forecast records for %s",
                        len(fetched),
                        municipality.dane_code,
                    )
                    # Re-read so the returned ORM objects include the new rows.
                    stored = (
                        db.query(MunicipalityClimateForecast)
                        .filter(
                            MunicipalityClimateForecast.municipality_dane_code == municipality.dane_code
                        )
                        .filter(MunicipalityClimateForecast.forecast_date >= today)
                        .filter(MunicipalityClimateForecast.forecast_date <= end_date)
                        .order_by(MunicipalityClimateForecast.forecast_date.asc())
                        .all()
                    )
            except Exception as exc:
                logger.warning(
                    "[get_forecast_records] Open-Meteo fetch failed for %s: %s",
                    municipality.dane_code,
                    exc,
                )

        return stored

    def _upsert_forecast_records(
        self,
        db: Session,
        municipality_dane_code: str,
        records: List[dict],
    ) -> None:
        """Upsert daily forecast records into the database."""
        now = datetime.now(timezone.utc)
        rows = []
        for record in records:
            rows.append(
                {
                    "municipality_dane_code": municipality_dane_code,
                    "forecast_date": record["forecast_date"],
                    "temp_min": record.get("temp_min"),
                    "temp_max": record.get("temp_max"),
                    "temp_mean": record.get("temp_mean"),
                    "precipitation": record.get("precipitation"),
                    "humidity": record.get("humidity"),
                    "uv_index": record.get("uv_index"),
                    "wind_speed": record.get("wind_speed"),
                    "fetched_at": now,
                    "updated_at": now,
                }
            )

        if not rows:
            return

        stmt = pg_insert(MunicipalityClimateForecast).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_climate_forecast",
            set_={
                "temp_min": stmt.excluded.temp_min,
                "temp_max": stmt.excluded.temp_max,
                "temp_mean": stmt.excluded.temp_mean,
                "precipitation": stmt.excluded.precipitation,
                "humidity": stmt.excluded.humidity,
                "uv_index": stmt.excluded.uv_index,
                "wind_speed": stmt.excluded.wind_speed,
                "fetched_at": stmt.excluded.fetched_at,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        db.execute(stmt)
        db.commit()

    def get_current_weather(
        self,
        db: Session,
        municipality: Municipality,
        max_age_minutes: int = 30,
    ) -> dict:
        """Return current weather for a municipality, using the DB as cache.

        If a cached record exists and is fresh (default < 30 minutes), it is
        returned. Otherwise, Open-Meteo is called and the result is persisted.
        """
        logger.debug(
            "[get_current_weather] Looking up cached current weather for %s",
            municipality.dane_code,
        )
        cached = (
            db.query(MunicipalityCurrentWeather)
            .filter(MunicipalityCurrentWeather.municipality_dane_code == municipality.dane_code)
            .first()
        )

        now = datetime.now(timezone.utc)
        if cached and cached.fetched_at and (now - cached.fetched_at).total_seconds() < max_age_minutes * 60:
            logger.info(
                "[get_current_weather] Returning cached current weather for %s (fetched_at=%s)",
                municipality.dane_code,
                cached.fetched_at,
            )
            return {
                "temperature": cached.temperature,
                "condition": cached.condition,
                "humidity": cached.humidity,
                "precipitation": cached.precipitation,
                "icon": cached.icon,
                "source": cached.source,
                "fetched_at": cached.fetched_at.isoformat(),
            }

        logger.info(
            "[get_current_weather] Fetching current weather for %s from Open-Meteo",
            municipality.dane_code,
        )
        weather_data = self.open_meteo.get_current_weather(
            lat=municipality.lat,
            lng=municipality.lng,
        )

        self._upsert_current_weather(db, municipality.dane_code, weather_data)
        return weather_data

    def _upsert_current_weather(
        self,
        db: Session,
        municipality_dane_code: str,
        weather_data: dict,
    ) -> None:
        """Upsert a current weather record."""
        now = datetime.now(timezone.utc)
        stmt = pg_insert(MunicipalityCurrentWeather).values(
            {
                "municipality_dane_code": municipality_dane_code,
                "temperature": weather_data["temperature"],
                "condition": weather_data["condition"],
                "humidity": weather_data["humidity"],
                "precipitation": weather_data["precipitation"],
                "icon": weather_data["icon"],
                "source": weather_data.get("source", "open-meteo"),
                "fetched_at": now,
                "updated_at": now,
            }
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_current_weather_municipality",
            set_={
                "temperature": stmt.excluded.temperature,
                "condition": stmt.excluded.condition,
                "humidity": stmt.excluded.humidity,
                "precipitation": stmt.excluded.precipitation,
                "icon": stmt.excluded.icon,
                "source": stmt.excluded.source,
                "fetched_at": stmt.excluded.fetched_at,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        db.execute(stmt)
        db.commit()
        logger.info(
            "[get_current_weather] Cached current weather for %s",
            municipality_dane_code,
        )
