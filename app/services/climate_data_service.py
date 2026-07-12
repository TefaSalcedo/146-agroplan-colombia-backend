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
from app.models import (
    Municipality,
    MunicipalityClimateForecast,
    MunicipalityCurrentWeather,
    MunicipalityMonthlyClimateForecast,
)
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

    def get_monthly_forecast_records(
        self,
        db: Session,
        municipality: Municipality,
        months: int,
        max_age_days: int = 30,
    ) -> List[MunicipalityMonthlyClimateForecast]:
        """Return monthly seasonal forecast records, fetching missing months from Open-Meteo.

        The seasonal forecast is updated once per month, so records are considered
        fresh while they are younger than ``max_age_days``. Missing months are fetched
        in a single Open-Meteo call and persisted.
        """
        today = self._today()
        requested_months = []
        cursor = today.replace(day=1)
        for _ in range(months):
            requested_months.append(cursor)
            if cursor.month == 12:
                cursor = cursor.replace(year=cursor.year + 1, month=1)
            else:
                cursor = cursor.replace(month=cursor.month + 1)

        logger.debug(
            "[get_monthly_forecast_records] Looking up stored forecasts for %s from %s",
            municipality.dane_code,
            requested_months,
        )

        stored = (
            db.query(MunicipalityMonthlyClimateForecast)
            .filter(
                MunicipalityMonthlyClimateForecast.municipality_dane_code == municipality.dane_code
            )
            .filter(MunicipalityMonthlyClimateForecast.forecast_month.in_(requested_months))
            .order_by(MunicipalityMonthlyClimateForecast.forecast_month.asc())
            .all()
        )

        # Determine which months are missing or stale.
        ttl = timedelta(days=max_age_days)
        now = datetime.now(timezone.utc)
        valid_by_month = {}
        for record in stored:
            if record.fetched_at and (now - record.fetched_at) <= ttl:
                valid_by_month[record.forecast_month] = record

        missing_months = [m for m in requested_months if m not in valid_by_month]

        logger.debug(
            "[get_monthly_forecast_records] Stored %s records, missing/stale %s months",
            len(valid_by_month),
            len(missing_months),
        )

        if missing_months:
            logger.info(
                "[get_monthly_forecast_records] Fetching %s monthly forecast months for %s from Open-Meteo",
                len(missing_months),
                municipality.dane_code,
            )
            try:
                # Ask for a slightly longer horizon so Open-Meteo returns complete
                # monthly aggregates for all requested months.
                fetched = self.open_meteo.get_monthly_seasonal_forecast(
                    lat=municipality.lat,
                    lng=municipality.lng,
                    months=months + 1,
                )
                if fetched:
                    self._upsert_monthly_forecast_records(db, municipality.dane_code, fetched)
                    logger.info(
                        "[get_monthly_forecast_records] Saved %s monthly forecast records for %s",
                        len(fetched),
                        municipality.dane_code,
                    )
                    # Re-read so returned ORM objects include new rows.
                    stored = (
                        db.query(MunicipalityMonthlyClimateForecast)
                        .filter(
                            MunicipalityMonthlyClimateForecast.municipality_dane_code
                            == municipality.dane_code
                        )
                        .filter(MunicipalityMonthlyClimateForecast.forecast_month.in_(requested_months))
                        .order_by(MunicipalityMonthlyClimateForecast.forecast_month.asc())
                        .all()
                    )
            except Exception as exc:
                logger.warning(
                    "[get_monthly_forecast_records] Open-Meteo fetch failed for %s: %s",
                    municipality.dane_code,
                    exc,
                )

        return stored

    def _upsert_monthly_forecast_records(
        self,
        db: Session,
        municipality_dane_code: str,
        records: List[dict],
    ) -> None:
        """Upsert monthly seasonal forecast records into the database."""
        now = datetime.now(timezone.utc)
        rows = []
        for record in records:
            forecast_month = record.get("forecast_month")
            if not forecast_month:
                continue

            temp_anomaly = record.get("temp_anomaly")
            precip_anomaly = record.get("precipitation_anomaly")
            trend_parts = []
            if temp_anomaly is not None:
                if temp_anomaly > 0.5:
                    trend_parts.append("warmer")
                elif temp_anomaly < -0.5:
                    trend_parts.append("cooler")
            if precip_anomaly is not None:
                if precip_anomaly > 10:
                    trend_parts.append("wetter")
                elif precip_anomaly < -10:
                    trend_parts.append("drier")
            trend = "_".join(trend_parts) if trend_parts else "neutral"

            rows.append(
                {
                    "municipality_dane_code": municipality_dane_code,
                    "forecast_month": forecast_month,
                    "temp_mean": record.get("temp_mean"),
                    "temp_anomaly": temp_anomaly,
                    "precipitation": record.get("precipitation"),
                    "precipitation_anomaly": precip_anomaly,
                    "trend": trend,
                    "source": record.get("source", "open-meteo-seasonal"),
                    "fetched_at": now,
                    "updated_at": now,
                }
            )

        if not rows:
            return

        stmt = pg_insert(MunicipalityMonthlyClimateForecast).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_monthly_climate_forecast",
            set_={
                "temp_mean": stmt.excluded.temp_mean,
                "temp_anomaly": stmt.excluded.temp_anomaly,
                "precipitation": stmt.excluded.precipitation,
                "precipitation_anomaly": stmt.excluded.precipitation_anomaly,
                "trend": stmt.excluded.trend,
                "source": stmt.excluded.source,
                "fetched_at": stmt.excluded.fetched_at,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        db.execute(stmt)
        db.commit()
