from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import get_settings
from app.database import SessionLocal
from app.models import ClimateSyncLog
from app.services.climate_sync import ClimateSyncService

settings = get_settings()

# Advisory lock key for climate sync (arbitrary fixed int)
_CLIMATE_SYNC_ADVISORY_LOCK = 20250101


def _create_db_session() -> Session:
    """Create a fresh database session for the background job."""
    return SessionLocal()


def _close_db_session(db: Session) -> None:
    """Close the database session safely."""
    try:
        db.close()
    except Exception:
        pass


def _run_sync_job(sync_type: str, days: int, cleanup: bool = False) -> None:
    """Execute a climate sync job in a background thread with advisory lock."""
    db = _create_db_session()
    try:
        # Try to acquire a session-level advisory lock
        result = db.execute(
            text("SELECT pg_try_advisory_lock(:lock_key)"),
            {"lock_key": _CLIMATE_SYNC_ADVISORY_LOCK},
        ).scalar()

        if not result:
            print("[climate_scheduler] Another sync job is already running. Skipping.")
            return

        try:
            service = ClimateSyncService(
                batch_size=settings.climate_sync_batch_size,
                delay_seconds=settings.climate_sync_delay_seconds,
            )
            service.run_sync(db, sync_type, days)

            if cleanup:
                service.delete_old_forecasts(db, keep_days=settings.climate_sync_cleanup_days)
        finally:
            db.execute(
                text("SELECT pg_advisory_unlock(:lock_key)"),
                {"lock_key": _CLIMATE_SYNC_ADVISORY_LOCK},
            )
    finally:
        _close_db_session(db)


def _offset_daily_time(hour: int, minute: int, offset_minutes: int) -> tuple[int, int]:
    """Safely add minutes to a daily schedule time, wrapping at 24h."""
    total = ((hour * 60) + minute + offset_minutes) % (24 * 60)
    return total // 60, total % 60


def get_scheduler() -> BackgroundScheduler:
    """Build and return a configured APScheduler instance."""
    scheduler = BackgroundScheduler()
    scheduler.configure(executors={"default": {"type": "threadpool", "max_workers": 1}})
    return scheduler


def schedule_climate_jobs(scheduler: BackgroundScheduler) -> None:
    """Schedule climate sync jobs if enabled."""
    if not settings.enable_climate_sync:
        return

    scheduler.add_job(
        _run_sync_job,
        trigger=CronTrigger(
            hour=settings.climate_sync_hour,
            minute=settings.climate_sync_minute,
        ),
        id="daily_climate_sync",
        name="Daily climate forecast refresh",
        replace_existing=True,
        args=["daily", 7, True],
    )

    weekly_hour, weekly_minute = _offset_daily_time(
        settings.climate_sync_hour,
        settings.climate_sync_minute,
        30,
    )
    scheduler.add_job(
        _run_sync_job,
        trigger=CronTrigger(day_of_week="sun", hour=weekly_hour, minute=weekly_minute),
        id="weekly_climate_extension",
        name="Weekly climate forecast extension",
        replace_existing=True,
        args=["weekly_extension", 7, False],
    )


def get_last_sync_status(db: Session) -> dict | None:
    """Return the most recent climate sync log entry."""
    log = db.query(ClimateSyncLog).order_by(ClimateSyncLog.started_at.desc()).first()
    if not log:
        return None

    return {
        "id": log.id,
        "sync_type": log.sync_type,
        "status": log.status,
        "started_at": log.started_at.isoformat() if log.started_at else None,
        "finished_at": log.finished_at.isoformat() if log.finished_at else None,
        "records_processed": log.records_processed,
        "records_failed": log.records_failed,
        "error_message": log.error_message,
    }
