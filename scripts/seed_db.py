import argparse
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from sqlalchemy import select
from app.database import SessionLocal
from app.config import get_settings
from app.models import (
    Department,
    Municipality,
    MunicipalityClimateForecast,
    ClimateSyncLog,
    Crop,
)
from app.services.climate_sync import ClimateSyncService

settings = get_settings()

# ---------------------------------------------------------------------------
# Crop definitions for the 7 ML-supported crops
# ---------------------------------------------------------------------------

SUPPORTED_CROPS = [
    {
        "id": "aguacate",
        "name": "Aguacate",
        "scientific_name": "Persea americana",
        "image": "/crops/aguacate.png",
        "days_to_harvest": 365,
        "soil_type": "Franco arenoso, profundo, bien drenado; pH 4.5 - 7.0",
        "ideal_temperature": "10 - 45 C",
        "humidity": "60 - 80 %",
        "precipitation": "300 - 2.500 mm / ano",
        "altitude": "0 - 2.800 msnm",
        "irrigation": "Moderado a alto; evitar encharcamiento",
        "substrates": ["Compost", "Materia organica"],
        "planting_months": [3, 4, 9, 10],
        "harvest_months": [6, 7, 8, 9, 10, 11, 12],
        "stages": [],
        "tips": [],
        "ml_crop_key": "aguacate",
        "is_ml_supported": True,
    },
    {
        "id": "algodon",
        "name": "Algodon",
        "scientific_name": "Gossypium hirsutum",
        "image": "/crops/algodon.png",
        "days_to_harvest": 200,
        "soil_type": "Franco a franco arcilloso, profundo, bien drenado; pH 5.0 - 9.5",
        "ideal_temperature": "15 - 42 C",
        "humidity": "50 - 70 %",
        "precipitation": "450 - 1.500 mm / ciclo",
        "altitude": "0 - 1.250 msnm",
        "irrigation": "Moderado; sensible al exceso de humedad",
        "substrates": ["Compost", "Estiercol curado"],
        "planting_months": [3, 4],
        "harvest_months": [8, 9, 10],
        "stages": [],
        "tips": [],
        "ml_crop_key": "algodon",
        "is_ml_supported": True,
    },
    {
        "id": "cana_panelera",
        "name": "Cana panelera",
        "scientific_name": "Saccharum officinarum",
        "image": "/crops/cana_panelera.png",
        "days_to_harvest": 365,
        "soil_type": "Franco, profundo, buena materia organica; pH 4.5 - 9.0",
        "ideal_temperature": "15 - 41 C",
        "humidity": "70 - 85 %",
        "precipitation": "1.000 - 5.000 mm / ano",
        "altitude": "0 - 1.600 msnm",
        "irrigation": "Moderado; tolera periodos secos cortos",
        "substrates": ["Compost", "Estiercol", "Cachaza"],
        "planting_months": [1, 2, 3, 4, 5, 6],
        "harvest_months": [10, 11, 12, 1, 2, 3],
        "stages": [],
        "tips": [],
        "ml_crop_key": "cana_panelera",
        "is_ml_supported": True,
    },
    {
        "id": "cebolla",
        "name": "Cebolla",
        "scientific_name": "Allium cepa",
        "image": "/crops/cebolla.png",
        "days_to_harvest": 150,
        "soil_type": "Franco arenoso, suelto y bien drenado; pH 4.3 - 8.3",
        "ideal_temperature": "4 - 30 C",
        "humidity": "60 - 70 %",
        "precipitation": "300 - 2.800 mm / ciclo",
        "altitude": "0 - 2.000 msnm",
        "irrigation": "Frecuente y ligero; sin encharcar",
        "substrates": ["Compost", "Humus de lombriz"],
        "planting_months": [3, 4, 9, 10],
        "harvest_months": [6, 7, 12, 1],
        "stages": [],
        "tips": [],
        "ml_crop_key": "cebolla",
        "is_ml_supported": True,
    },
    {
        "id": "fresa",
        "name": "Fresa",
        "scientific_name": "Fragaria x ananassa",
        "image": "/crops/fresa.png",
        "days_to_harvest": 270,
        "soil_type": "Franco arenoso, rico en materia organica; pH 4.5 - 6.5",
        "ideal_temperature": "5 - 28 C",
        "humidity": "65 - 75 %",
        "precipitation": "300 - 2.800 mm / ano",
        "altitude": "0 - 2.000 msnm",
        "irrigation": "Frecuente, por goteo preferentemente",
        "substrates": ["Compost", "Cascarilla de arroz"],
        "planting_months": [3, 4, 9, 10],
        "harvest_months": [6, 7, 10, 11, 12],
        "stages": [],
        "tips": [],
        "ml_crop_key": "fresa",
        "is_ml_supported": True,
    },
    {
        "id": "pina",
        "name": "Pina",
        "scientific_name": "Ananas comosus",
        "image": "/crops/pina.png",
        "days_to_harvest": 365,
        "soil_type": "Franco arenoso, ligeramente acido, bien drenado; pH 3.5 - 9.0",
        "ideal_temperature": "10 - 36 C",
        "humidity": "70 - 85 %",
        "precipitation": "550 - 3.500 mm / ano",
        "altitude": "0 - 1.800 msnm",
        "irrigation": "Moderado; tolera sequias cortas",
        "substrates": ["Compost", "Materia organica"],
        "planting_months": [3, 4, 5, 6],
        "harvest_months": [3, 4, 5, 6, 7],
        "stages": [],
        "tips": [],
        "ml_crop_key": "pina",
        "is_ml_supported": True,
    },
    {
        "id": "soya",
        "name": "Soya",
        "scientific_name": "Glycine max",
        "image": "/crops/soya.png",
        "days_to_harvest": 180,
        "soil_type": "Franco, bien drenado; pH 5.5 - 7.0",
        "ideal_temperature": "10 - 38 C",
        "humidity": "50 - 70 %",
        "precipitation": "450 - 1.800 mm / ciclo",
        "altitude": "0 - 3.000 msnm",
        "irrigation": "Moderado; critico en floracion y llenado de vaina",
        "substrates": ["Compost", "Inoculo de Rhizobium"],
        "planting_months": [3, 4, 9, 10],
        "harvest_months": [6, 7, 12, 1],
        "stages": [],
        "tips": [],
        "ml_crop_key": "soya",
        "is_ml_supported": True,
    },
]


def _resolve_municipality_json_path() -> str | None:
    json_path = "data/divipola_municipios.json"
    if os.path.exists(json_path):
        return json_path

    fallback_path = os.path.join(settings.ml_data_path, "data/divipola_municipios.json")
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
    department_count = len({row["cod_dpto"] for row in data})
    return municipality_count, department_count


def validate_municipality_coverage(db: Session) -> tuple[bool, dict]:
    source_data, source_path = _load_municipality_source()
    expected_municipalities, expected_departments = _source_stats(source_data)

    db_municipalities = db.query(Municipality).count()
    db_departments = db.query(Department).count()

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
    """Fetch and store climate forecasts for municipalities."""
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
    """Convert coordinate from comma decimal to point decimal format."""
    return float(coord_str.replace(",", "."))


def seed_departments(db: Session, data: list[dict]) -> tuple[int, int]:
    """Upsert departments from the municipality source data."""
    created = 0
    updated = 0
    seen_codes: set[str] = set()

    for row in data:
        dane_code = row["cod_dpto"]
        if dane_code in seen_codes:
            continue
        seen_codes.add(dane_code)

        existing = db.query(Department).filter(Department.dane_code == dane_code).first()
        if existing:
            if existing.name != row["dpto"]:
                existing.name = row["dpto"]
                updated += 1
            continue

        db.add(Department(dane_code=dane_code, name=row["dpto"]))
        created += 1

    db.flush()
    return created, updated


def seed_municipalities(
    db: Session,
    data: list[dict],
    complete: bool = False,
) -> tuple[int, int]:
    """Upsert municipalities using cod_mpio as the DANE code (5 digits)."""
    created = 0
    updated = 0

    for row in data:
        dane_code = row["cod_mpio"]
        department_code = row["cod_dpto"]

        existing = db.query(Municipality).filter(Municipality.dane_code == dane_code).first()
        payload = {
            "dane_code": dane_code,
            "department_dane_code": department_code,
            "name": row["nom_mpio"],
            "municipality_type": row.get("tipo_municipio"),
            "lat": convert_coordinate(row["latitud"]),
            "lng": convert_coordinate(row["longitud"]),
            "altitude": 0,
            "avg_temperature": 0.0,
            "precipitation": 0.0,
        }

        if existing:
            if complete:
                for key, value in payload.items():
                    setattr(existing, key, value)
                updated += 1
            continue

        db.add(Municipality(**payload))
        created += 1

    db.flush()
    return created, updated


def seed_crops(db: Session) -> tuple[int, int]:
    """Upsert the 7 ML-supported crops."""
    created = 0
    updated = 0

    for crop_data in SUPPORTED_CROPS:
        crop_id = crop_data["id"]
        existing = db.query(Crop).filter(Crop.id == crop_id).first()

        if existing:
            for key, value in crop_data.items():
                setattr(existing, key, value)
            updated += 1
            continue

        db.add(Crop(**crop_data))
        created += 1

    db.flush()
    return created, updated


def seed_all(
    force: bool = False,
    limit: int | None = None,
    complete: bool = False,
    strict_validation: bool = False,
    skip_crops: bool = False,
):
    """Seed departments, municipalities, and crops from source data."""
    db: Session = SessionLocal()

    try:
        data, source_path = _load_municipality_source()
        full_expected_municipalities, full_expected_departments = _source_stats(data)

        if limit:
            data = data[:limit]

        if force:
            existing_count = db.query(Municipality).count()
            if existing_count > 0:
                print(f"Deleting {existing_count} existing municipality records...")
                db.query(MunicipalityClimateForecast).delete()
                db.query(Municipality).delete()
                db.commit()

            dept_count = db.query(Department).count()
            if dept_count > 0:
                print(f"Deleting {dept_count} existing department records...")
                db.query(Department).delete()
                db.commit()

        # Seed departments first
        dept_created, dept_updated = seed_departments(db, data)
        print(f"Departments - Created: {dept_created}, Updated: {dept_updated}")

        # Seed municipalities
        muni_created, muni_updated = seed_municipalities(db, data, complete=complete)
        print(f"Municipalities - Created: {muni_created}, Updated: {muni_updated}")

        # Seed crops
        if not skip_crops:
            crop_created, crop_updated = seed_crops(db)
            print(f"Crops - Created: {crop_created}, Updated: {crop_updated}")

        db.commit()

        total_muni = db.query(Municipality).count()
        total_dept = db.query(Department).count()
        total_crops = db.query(Crop).count()
        print(f"Source used: {source_path}")
        print(f"DB totals: {total_muni} municipalities, {total_dept} departments, {total_crops} crops")

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
        print(f"Error seeding database: {e}")
        raise
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Seed the AgroPlan database (requires Alembic migrations to be applied first)")
    parser.add_argument("--force", action="store_true", help="Delete existing records before seeding")
    parser.add_argument("--complete", action="store_true", help="Update existing records without deleting")
    parser.add_argument("--validate-only", action="store_true", help="Validate DB coverage against source JSON")
    parser.add_argument("--strict", action="store_true", help="Fail if coverage does not match expected source coverage")
    parser.add_argument("--allow-partial", action="store_true", help="Allow partial dataset loads when using --limit")
    parser.add_argument("--skip-crops", action="store_true", help="Skip crop seeding")
    parser.add_argument("--sync-climate", action="store_true", help="Fetch and store climate forecasts after seeding")
    parser.add_argument("--limit", type=int, default=None, help="Only seed the first N municipalities")
    parser.add_argument("--sync-limit", type=int, default=None, help="Only sync climate for the first N municipalities")
    parser.add_argument("--days", type=int, default=None, help="Forecast days to fetch (default: CLIMATE_SYNC_DAYS_AHEAD)")
    parser.add_argument("--delay", type=float, default=None, help="Seconds between Open-Meteo requests")
    parser.add_argument("--retries", type=int, default=1, help="Retries per Open-Meteo request")
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

    seed_all(
        force=args.force,
        limit=args.limit,
        complete=args.complete,
        strict_validation=args.strict,
        skip_crops=args.skip_crops,
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
