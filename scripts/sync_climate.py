"""Manual runner for the Open-Meteo climate sync job.

Adds the project root to sys.path so it can be run both inside and outside
of a container.

Usage:
    python scripts/sync_climate.py [--days-ahead 16] [--limit 10]

Example (inside the API container):
    docker compose exec api python scripts/sync_climate.py --days-ahead 16
"""

import argparse
import sys
from pathlib import Path

# Make the project root importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal
from app.services.climate_sync import ClimateSyncService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Open-Meteo climate sync manually")
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=16,
        help="Number of days of forecast to fetch (default: 16)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit sync to the first N municipalities (default: all)",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        service = ClimateSyncService()
        log = service.run_sync(
            db=db,
            sync_type="manual",
            days=args.days_ahead,
            municipality_limit=args.limit,
        )
        print(
            f"[sync_climate] Finished: status={log.status}, "
            f"records={log.records_processed}, failed={log.records_failed}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
