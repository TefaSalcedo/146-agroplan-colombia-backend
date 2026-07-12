import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import List

from sqlalchemy.orm import Session
from sqlalchemy import delete, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import get_settings
from app.models import Municipality, MunicipalityClimateForecast, ClimateSyncLog
from app.services.open_meteo import OpenMeteoService


def _utcnow() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class ClimateSyncService:
    """Background service that fetches and stores Open-Meteo forecasts for municipalities."""

    def __init__(
        self,
        batch_size: int | None = None,
        delay_seconds: float | None = None,
        max_retries: int = 3,
    ):
        settings = get_settings()
        self.weather_service = OpenMeteoService()
        self.batch_size = batch_size if batch_size is not None else settings.climate_sync_batch_size
        self.delay_seconds = (
            delay_seconds if delay_seconds is not None else settings.climate_sync_delay_seconds
        )
        self.max_retries = max_retries

    def run_sync(
        self,
        db: Session,
        sync_type: str,
        days: int,
        municipality_limit: int | None = None,
    ) -> ClimateSyncLog:
        """Run a climate sync job synchronously."""
        print(
            f"[climate_sync] Starting {sync_type} sync for "
            f"{municipality_limit or 'all'} municipalities, {days} days forecast"
        )
        log = ClimateSyncLog(
            sync_type=sync_type,
            started_at=_utcnow(),
            records_processed=0,
            records_failed=0,
            status="running",
        )
        db.add(log)
        db.commit()
        db.refresh(log)

        try:
            municipalities = db.query(Municipality).order_by(Municipality.dane_code).all()
            if municipality_limit:
                municipalities = municipalities[:municipality_limit]
            total_records = 0
            total_failed = 0

            for batch_start in range(0, len(municipalities), self.batch_size):
                batch = municipalities[batch_start : batch_start + self.batch_size]
                print(
                    f"[climate_sync] Processing batch {batch_start // self.batch_size + 1} "
                    f"({len(batch)} municipalities)"
                )
                batch_records, failed_count = asyncio.run(self._sync_batch(batch, days))

                for municipality_id, records in batch_records:
                    self._upsert_records(db, municipality_id, records)
                    total_records += len(records)

                total_failed += failed_count
                db.commit()

                if batch_start + self.batch_size < len(municipalities):
                    time.sleep(self.delay_seconds)

            log.finished_at = _utcnow()
            log.records_processed = total_records
            log.records_failed = total_failed
            log.status = "success" if total_failed == 0 else "partial"
            db.commit()

        except Exception as e:
            log.finished_at = _utcnow()
            log.status = "error"
            log.error_message = str(e)
            db.commit()
            raise

        return log

    async def _sync_batch(
        self,
        municipalities: List[Municipality],
        days: int,
    ) -> tuple[List[tuple[str, List[dict]]], int]:
        """Fetch forecasts for a batch of municipalities with retries and throttling."""
        results: List[tuple[str, List[dict]]] = []
        failed_count = 0

        for index, municipality in enumerate(municipalities):
            print(
                f"[climate_sync] {index + 1}/{len(municipalities)} "
                f"{municipality.name} ({municipality.dane_code})"
            )
            records = await self._fetch_with_retry(municipality.lat, municipality.lng, days)

            if records is not None:
                results.append((municipality.dane_code, records))
            else:
                failed_count += 1

            await asyncio.sleep(self.delay_seconds)

        return results, failed_count

    async def _fetch_with_retry(
        self,
        lat: float,
        lng: float,
        days: int,
    ) -> List[dict] | None:
        """Fetch forecast with exponential backoff retries."""
        for attempt in range(self.max_retries):
            try:
                return await self.weather_service.get_daily_forecast(lat, lng, days)
            except Exception:
                if attempt == self.max_retries - 1:
                    return None
                await asyncio.sleep(2**attempt)

        return None

    def _upsert_records(self, db: Session, municipality_dane_code: str, records: List[dict]) -> None:
        """Upsert daily forecast records using PostgreSQL native ON CONFLICT."""
        now = _utcnow()
        rows = []
        for record in records:
            rows.append({
                "municipality_dane_code": municipality_dane_code,
                "forecast_date": record["forecast_date"],
                "temp_min": record.get("temp_min"),
                "temp_max": record.get("temp_max"),
                "temp_mean": record.get("temp_mean"),
                "precipitation": record.get("precipitation"),
                "humidity": record.get("humidity"),
                "uv_index": record.get("uv_index"),
                "wind_speed": record.get("wind_speed"),
                "updated_at": now,
            })

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
                "updated_at": stmt.excluded.updated_at,
            },
        )
        db.execute(stmt)

    def delete_old_forecasts(self, db: Session, keep_days: int = 180) -> int:
        """Delete forecasts older than keep_days to avoid unbounded table growth."""
        cutoff = _utcnow().date() - timedelta(days=keep_days)
        result = db.execute(
            delete(MunicipalityClimateForecast).where(
                MunicipalityClimateForecast.forecast_date < cutoff
            )
        )
        db.commit()
        return result.rowcount


def run_climate_sync(db: Session, sync_type: str, days: int) -> ClimateSyncLog:
    """Convenience function to run a climate sync with default settings."""
    service = ClimateSyncService()
    return service.run_sync(db, sync_type, days)
