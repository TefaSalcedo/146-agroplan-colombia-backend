import argparse
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from app.database import engine, SessionLocal, Base
from app.config import get_settings
from app.models import Municipality, MunicipalityClimateForecast, ClimateSyncLog
from app.services.climate_sync import ClimateSyncService

settings = get_settings()


def create_tables():
    """Create all tables if they don't exist"""
    Base.metadata.create_all(bind=engine)


def _resolve_municipality_json_path() -> str | None:
    json_path = "data/divipola_municipios.json"
    if os.path.exists(json_path):
        return json_path

    fallback_path = os.path.join(settings.ml_data_path, "../data/divipola_municipios.json")
    if os.path.exists(fallback_path):
        return fallback_path

    return None


def _load_municipality_source() -> tuple[list[dict], str]:
    json_path = _resolve_municipality_json_path()
    if not json_path:
        raise FileNotFoundError("Municipality JSON source not found")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data, json_path


def _source_stats(data: list[dict]) -> tuple[int, int]:
    municipality_count = len(data)
    department_count = len({row["dpto"] for row in data})
    return municipality_count, department_count


def validate_municipality_coverage(db: Session) -> tuple[bool, dict]:
    source_data, source_path = _load_municipality_source()
    expected_municipalities, expected_departments = _source_stats(source_data)

    db_municipalities = db.query(Municipality).count()
    db_departments = db.query(Municipality.department).distinct().count()

    is_valid = (
        db_municipalities >= expected_municipalities
        and db_departments >= expected_departments
    )

    details = {
        "source_path": source_path,
        "expected_municipalities": expected_municipalities,
        "expected_departments": expected_departments,
        "db_municipalities": db_municipalities,
        "db_departments": db_departments,
    }
    return is_valid, details


def sync_climate_forecasts(
    days: int | None = None,
    municipality_limit: int | None = None,
    delay_seconds: float | None = None,
    max_retries: int = 1,
) -> None:
    """Fetch and store climate forecasts for municipalities.

    Defaults are read from environment settings when not provided.
    """
    create_tables()

    if days is None:
        days = settings.climate_sync_days_ahead
    if delay_seconds is None:
        delay_seconds = settings.climate_sync_delay_seconds

    db = SessionLocal()
    try:
        service = ClimateSyncService(
            delay_seconds=delay_seconds,
            max_retries=max_retries,
        )
        service.run_sync(db, "initial", days, municipality_limit=municipality_limit)
    finally:
        db.close()


def convert_coordinate(coord_str: str) -> float:
    """Convert coordinate from comma decimal to point decimal format"""
    return float(coord_str.replace(",", "."))


def seed_municipalities(
    force: bool = False,
    limit: int | None = None,
    complete: bool = False,
    strict_validation: bool = False,
):
    """Seed municipalities from JSON file"""
    create_tables()

    db: Session = SessionLocal()

    try:
        data, source_path = _load_municipality_source()
        full_expected_municipalities, full_expected_departments = _source_stats(data)

        if limit:
            data = data[:limit]

        existing_count = db.query(Municipality).count()
        if existing_count > 0 and not force and not complete:
            print(f"Municipalities table already has {existing_count} records. Skipping seed.")
            return

        if force and existing_count > 0:
            print(f"Deleting {existing_count} existing municipality records...")
            db.query(Municipality).delete()
            db.commit()

        created = 0
        updated = 0

        for row in data:
            municipality_id = f"{row['cod_dpto']}{row['cod_mpio']}"
            payload = {
                "id": municipality_id,
                "name": row["nom_mpio"],
                "department": row["dpto"],
                "lat": convert_coordinate(row["latitud"]),
                "lng": convert_coordinate(row["longitud"]),
                "altitude": 0,
                "avg_temperature": 0.0,
                "precipitation": 0.0,
                "dane_code": f"{row['cod_dpto']}{row['cod_mpio']}",
            }

            existing = db.query(Municipality).filter(Municipality.id == municipality_id).first()
            if existing:
                if complete:
                    existing.name = payload["name"]
                    existing.department = payload["department"]
                    existing.lat = payload["lat"]
                    existing.lng = payload["lng"]
                    existing.altitude = payload["altitude"]
                    existing.avg_temperature = payload["avg_temperature"]
                    existing.precipitation = payload["precipitation"]
                    existing.dane_code = payload["dane_code"]
                    updated += 1
                continue

            db.add(Municipality(**payload))
            created += 1

        db.commit()

        total = db.query(Municipality).count()
        departments = db.query(Municipality.department).distinct().count()
        print(f"Source used: {source_path}")
        print(f"Created: {created}, Updated: {updated}")
        print(f"Municipalities in DB: {total}")
        print(f"Departments in DB: {departments}")

        if strict_validation:
            is_valid, details = validate_municipality_coverage(db)
            if not is_valid:
                raise RuntimeError(
                    "Coverage validation failed. "
                    f"Expected >= {details['expected_municipalities']} municipalities and "
                    f">= {details['expected_departments']} departments, got "
                    f"{details['db_municipalities']} municipalities and "
                    f"{details['db_departments']} departments."
                )
            print(
                "Coverage validation passed: "
                f"{details['db_municipalities']} municipalities / "
                f"{details['db_departments']} departments "
                f"(source: {details['expected_municipalities']} / {details['expected_departments']})"
            )

        if not limit:
            print(
                "Expected full source coverage: "
                f"{full_expected_municipalities} municipalities / {full_expected_departments} departments"
            )

    except Exception as e:
        db.rollback()
        print(f"Error seeding municipalities: {e}")
        raise
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Seed the local AgroPlan database")
    parser.add_argument("--force", action="store_true", help="Delete existing records before seeding")
    parser.add_argument("--complete", action="store_true", help="Complete/update municipalities without deleting existing records")
    parser.add_argument("--validate-only", action="store_true", help="Validate DB municipality/department coverage against source JSON")
    parser.add_argument("--strict", action="store_true", help="Fail if coverage does not match expected source coverage")
    parser.add_argument("--allow-partial", action="store_true", help="Allow partial dataset loads when using --limit")
    parser.add_argument("--sync-climate", action="store_true", help="Fetch and store climate forecasts after seeding")
    parser.add_argument("--limit", type=int, default=None, help="Only seed the first N municipalities")
    parser.add_argument("--sync-limit", type=int, default=None, help="Only sync climate for the first N municipalities")
    parser.add_argument("--days", type=int, default=None, help="Forecast days to fetch (default: CLIMATE_SYNC_DAYS_AHEAD or 90)")
    parser.add_argument("--delay", type=float, default=None, help="Seconds between Open-Meteo requests (default: CLIMATE_SYNC_DELAY_SECONDS or 2.0)")
    parser.add_argument("--retries", type=int, default=1, help="Retries per Open-Meteo request (default: 1)")
    args = parser.parse_args()

    if args.limit is not None and not args.allow_partial:
        parser.error("--limit creates a partial dataset. Add --allow-partial to confirm intentionally partial load.")

    if args.complete and args.force:
        parser.error("Use either --complete or --force, not both.")

    if args.complete and args.limit is not None:
        parser.error("--complete requires full source load. Remove --limit.")

    if args.validate_only:
        db = SessionLocal()
        try:
            is_valid, details = validate_municipality_coverage(db)
            print(f"Source path: {details['source_path']}")
            print(
                "Expected source coverage: "
                f"{details['expected_municipalities']} municipalities / {details['expected_departments']} departments"
            )
            print(
                "DB coverage: "
                f"{details['db_municipalities']} municipalities / {details['db_departments']} departments"
            )
            if not is_valid:
                raise SystemExit(1)
            return
        finally:
            db.close()

    seed_municipalities(
        force=args.force,
        limit=args.limit,
        complete=args.complete,
        strict_validation=args.strict,
    )

    if args.sync_climate:
        print("Starting climate forecast sync...")
        sync_climate_forecasts(
            days=args.days,
            municipality_limit=args.sync_limit or args.limit,
            delay_seconds=args.delay,
            max_retries=args.retries,
        )


if __name__ == "__main__":
    main()
