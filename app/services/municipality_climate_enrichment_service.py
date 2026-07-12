"""Service for enriching municipality climate data from Open-Meteo.

When a municipality has missing altitude, temperature or precipitation values,
this service fetches them from Open-Meteo and stores them in a dedicated table.
The enriched data expires monthly so it can be refreshed for temperature and
precipitation while keeping the stable altitude value.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.logger import get_logger
from app.models import Municipality, MunicipalityClimateEnrichment
from app.services.open_meteo import OpenMeteoService

logger = get_logger("app.services.municipality_climate_enrichment_service")

ENRICHMENT_TTL_DAYS = 30


class MunicipalityClimateEnrichmentService:
    """Fetch and cache municipality climate data from Open-Meteo."""

    def __init__(self):
        self.open_meteo = OpenMeteoService()

    def get_enriched_data(
        self,
        db: Session,
        municipality: Municipality,
        force: bool = False,
    ) -> MunicipalityClimateEnrichment:
        """Return enriched climate data for a municipality.

        Uses the cached row if it exists and is still valid. Otherwise fetches
        altitude and annual climate from Open-Meteo and stores the result.
        """
        now = datetime.now(timezone.utc)
        dane = municipality.dane_code

        cached = (
            db.query(MunicipalityClimateEnrichment)
            .filter(MunicipalityClimateEnrichment.municipality_dane_code == dane)
            .first()
        )

        if cached and not force:
            if cached.expires_at and cached.expires_at > now:
                logger.info(
                    "[get_enriched_data] Returning cached enrichment for municipality=%s (expires_at=%s)",
                    dane,
                    cached.expires_at,
                )
                return cached
            logger.info(
                "[get_enriched_data] Enrichment expired for municipality=%s, refreshing",
                dane,
            )
        else:
            logger.info(
                "[get_enriched_data] No enrichment for municipality=%s, fetching from Open-Meteo",
                dane,
            )

        altitude = self.open_meteo.get_elevation(municipality.lat, municipality.lng)
        climate = self.open_meteo.get_annual_climate(municipality.lat, municipality.lng)

        expires_at = now + timedelta(days=ENRICHMENT_TTL_DAYS)

        if cached:
            # Keep previously fetched altitude if the new call failed.
            if altitude is not None:
                cached.altitude = int(altitude)
            if climate is not None:
                cached.avg_temperature = climate.get("avg_temperature")
                cached.precipitation = climate.get("precipitation")
            cached.fetched_at = now
            cached.expires_at = expires_at
        else:
            cached = MunicipalityClimateEnrichment(
                municipality_dane_code=dane,
                altitude=int(altitude) if altitude is not None else None,
                avg_temperature=climate.get("avg_temperature") if climate else None,
                precipitation=climate.get("precipitation") if climate else None,
                source="open-meteo",
                fetched_at=now,
                expires_at=expires_at,
            )
            db.add(cached)

        db.commit()
        db.refresh(cached)
        logger.info(
            "[get_enriched_data] Saved enrichment for municipality=%s (altitude=%s, temp=%s, precip=%s, expires_at=%s)",
            dane,
            cached.altitude,
            cached.avg_temperature,
            cached.precipitation,
            expires_at,
        )
        return cached

    def get_effective_values(
        self,
        db: Session,
        municipality: Municipality,
    ) -> dict:
        """Return the best available altitude, temperature and precipitation.

        Prefer the official municipality values when they are present. Otherwise
        use the Open-Meteo enrichment data.
        """
        enrichment = self.get_enriched_data(db, municipality)

        altitude = municipality.altitude or enrichment.altitude
        avg_temperature = municipality.avg_temperature or enrichment.avg_temperature
        precipitation = municipality.precipitation or enrichment.precipitation

        # Treat zero as missing for these fields.
        if not altitude:
            altitude = enrichment.altitude
        if not avg_temperature:
            avg_temperature = enrichment.avg_temperature
        if not precipitation:
            precipitation = enrichment.precipitation

        return {
            "altitude": altitude,
            "avg_temperature": avg_temperature,
            "precipitation": precipitation,
            "source": "municipality" if municipality.altitude and municipality.avg_temperature and municipality.precipitation else "open-meteo-enrichment",
        }


# Singleton
_enrichment_service: Optional[MunicipalityClimateEnrichmentService] = None


def get_municipality_climate_enrichment_service() -> MunicipalityClimateEnrichmentService:
    global _enrichment_service
    if _enrichment_service is None:
        _enrichment_service = MunicipalityClimateEnrichmentService()
    return _enrichment_service
