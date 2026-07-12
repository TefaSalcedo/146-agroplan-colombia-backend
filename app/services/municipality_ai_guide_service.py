"""Service for generating and caching municipality AI guides.

Guides are generated with an LLM and cached for 3 months. If a fresh guide
exists in the database it is returned directly; otherwise a new one is
generated and persisted.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.models import (
    Crop,
    Municipality,
    MunicipalityAIGuide,
    MunicipalityClimateForecast,
    MunicipalityCurrentWeather,
    MunicipalityMonthlyClimateForecast,
)
from app.logger import get_logger
from app.services.crop_catalog import CropCatalog
from app.services.llm_service import get_llm_service, log_llm_generation, PROMPT_SCHEMA_VERSION
from app.services.municipality_climate_enrichment_service import (
    get_municipality_climate_enrichment_service,
)

logger = get_logger("app.services.municipality_ai_guide_service")

GUIDE_VERSION = "2.0"
GUIDE_TTL_DAYS = 90

# Candidate locations for the integrated crop-features dataset.
_CROP_FEATURES_PATHS = [
    Path("models/yield/data/crop_features_integrated.csv"),
    Path("data/crop_features_integrated.csv"),
    Path("/app/models/yield/data/crop_features_integrated.csv"),
]


class MunicipalityAIGuideService:
    """Manage LLM-generated municipality guides with 3-month caching."""

    def __init__(self):
        self.llm_service = get_llm_service()
        self.crop_catalog = CropCatalog()
        self.enrichment_service = get_municipality_climate_enrichment_service()
        self._crop_features: Optional[pd.DataFrame] = None

    def _load_crop_features(self) -> Optional[pd.DataFrame]:
        """Load the integrated crop-features dataset once."""
        if self._crop_features is not None:
            return self._crop_features

        for path in _CROP_FEATURES_PATHS:
            if path.exists():
                try:
                    self._crop_features = pd.read_csv(str(path))
                    logger.info(
                        "[load_crop_features] Loaded crop features from %s (%s rows)",
                        path,
                        len(self._crop_features),
                    )
                    return self._crop_features
                except Exception as e:
                    logger.warning("[load_crop_features] Could not load %s: %s", path, e)

        logger.warning("[load_crop_features] No crop-features dataset found")
        return None

    def _filter_compatible_crops(
        self,
        df: pd.DataFrame,
        altitude: Optional[float],
        avg_temperature: Optional[float],
        precipitation: Optional[float],
        exclude_names: List[str],
        top_n: int = 15,
    ) -> List[Dict[str, Any]]:
        """Return crops from EcoCrop/FAO that match the municipality climate.

        Filters by temperature, precipitation and altitude ranges when available.
        Excludes crops already present in the AgroPlan catalog. Returns the best
        matches up to ``top_n`` rows.
        """
        if df is None or df.empty:
            return []

        # Work on a copy and keep only rows with a name.
        rows = df[df["cultivo_eva"].notna()].copy()

        def _to_float(value):
            try:
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return None
                return float(value)
            except (ValueError, TypeError):
                return None

        # Temperature filter.
        if avg_temperature is not None:
            rows["temp_min"] = rows["temp_opt_min"].apply(_to_float)
            rows["temp_max"] = rows["temp_opt_max"].apply(_to_float)
            mask = (
                (rows["temp_min"].isna()) | (rows["temp_min"] <= avg_temperature)
            ) & (
                (rows["temp_max"].isna()) | (rows["temp_max"] >= avg_temperature)
            )
            rows = rows[mask].copy()

        # Precipitation filter.
        if precipitation is not None:
            rows["rain_min"] = rows["rain_opt_min"].apply(_to_float)
            rows["rain_max"] = rows["rain_opt_max"].apply(_to_float)
            mask = (
                (rows["rain_min"].isna()) | (rows["rain_min"] <= precipitation)
            ) & (
                (rows["rain_max"].isna()) | (rows["rain_max"] >= precipitation)
            )
            rows = rows[mask].copy()

        # Altitude filter (alt_max is the maximum altitude the crop tolerates).
        if altitude is not None:
            rows["alt_max"] = rows["alt_max"].apply(_to_float)
            mask = (rows["alt_max"].isna()) | (rows["alt_max"] >= altitude)
            rows = rows[mask].copy()

        # Exclude crops already in the AgroPlan catalog.
        exclude_normalized = {name.lower().strip() for name in exclude_names if name}
        rows = rows[~rows["cultivo_eva"].str.lower().isin(exclude_normalized)].copy()

        if rows.empty:
            return []

        # Prefer rows with full EcoCrop data, then sort by name for stability.
        rows["has_ecocrop"] = rows["has_ecocrop"].fillna(False)
        rows = rows.sort_values(
            by=["has_ecocrop", "ecocrop_match_score", "cultivo_eva"],
            ascending=[False, False, True],
        )

        selected = rows.head(top_n)
        result = []
        for _, row in selected.iterrows():
            result.append({
                "crop_name": str(row.get("cultivo_eva", "")).strip(),
                "scientific_name": str(row.get("ecocrop_scientific_name", "")).strip() or None,
                "category": str(row.get("ecocrop_category", "")).strip() or None,
                "cycle_days_min": _to_float(row.get("gmin_dias")),
                "cycle_days_max": _to_float(row.get("gmax_dias")),
                "temp_optimal_range": str(row.get("temp_optimal_range", "")).strip() or None,
                "rain_optimal_range": str(row.get("rain_optimal_range", "")).strip() or None,
                "altitude_max": _to_float(row.get("alt_max")),
                "sowing_months": str(row.get("fao_sowing_months", "")).strip() or None,
            })

        return result

    def _get_current_weather(self, db: Session, municipality_id: str) -> Optional[Dict[str, Any]]:
        """Return current weather for the municipality if available."""
        row = (
            db.query(MunicipalityCurrentWeather)
            .filter(MunicipalityCurrentWeather.municipality_dane_code == municipality_id)
            .first()
        )
        if not row:
            return None
        return {
            "temperature": row.temperature,
            "condition": row.condition,
            "humidity": row.humidity,
            "precipitation": row.precipitation,
        }

    def _get_daily_forecast_summary(self, db: Session, municipality_id: str) -> Optional[Dict[str, Any]]:
        """Return a short summary of the next 7 daily forecast records."""
        rows = (
            db.query(MunicipalityClimateForecast)
            .filter(MunicipalityClimateForecast.municipality_dane_code == municipality_id)
            .order_by(MunicipalityClimateForecast.forecast_date)
            .limit(7)
            .all()
        )
        if not rows:
            return None
        temps = [r.temp_mean for r in rows if r.temp_mean is not None]
        precips = [r.precipitation for r in rows if r.precipitation is not None]
        return {
            "next_days_count": len(rows),
            "temp_mean_range": [round(min(temps), 1), round(max(temps), 1)] if temps else None,
            "precipitation_total_mm": round(sum(precips), 1) if precips else None,
        }

    def _get_monthly_forecast_summary(self, db: Session, municipality_id: str) -> Optional[List[Dict[str, Any]]]:
        """Return the monthly seasonal outlook records."""
        rows = (
            db.query(MunicipalityMonthlyClimateForecast)
            .filter(MunicipalityMonthlyClimateForecast.municipality_dane_code == municipality_id)
            .order_by(MunicipalityMonthlyClimateForecast.forecast_month)
            .all()
        )
        if not rows:
            return None
        return [
            {
                "month": r.forecast_month.isoformat(),
                "temp_mean": r.temp_mean,
                "precipitation": r.precipitation,
                "trend": r.trend,
            }
            for r in rows
        ]

    def _build_context(
        self,
        db: Session,
        municipality: Municipality,
    ) -> Dict[str, Any]:
        """Build the context object sent to the LLM."""
        effective = self.enrichment_service.get_effective_values(db, municipality)

        ml_crops = self.crop_catalog.get_ml_supported_crops(db)
        catalog_crop_names = [c.name for c in ml_crops]
        all_crops = db.query(Crop).order_by(Crop.name).all()
        all_crop_names = [c.name for c in all_crops]

        df = self._load_crop_features()
        alternative_crops = self._filter_compatible_crops(
            df,
            altitude=effective.get("altitude"),
            avg_temperature=effective.get("avg_temperature"),
            precipitation=effective.get("precipitation"),
            exclude_names=all_crop_names,
            top_n=15,
        )

        return {
            "municipality_id": municipality.dane_code,
            "municipality_name": municipality.name,
            "department_name": municipality.department or "",
            "latitude": municipality.lat,
            "longitude": municipality.lng,
            "altitude_meters": effective.get("altitude"),
            "avg_temperature_c": effective.get("avg_temperature"),
            "annual_precipitation_mm": effective.get("precipitation"),
            "climate_source": effective.get("source"),
            "current_weather": self._get_current_weather(db, municipality.dane_code),
            "daily_forecast_summary": self._get_daily_forecast_summary(db, municipality.dane_code),
            "monthly_forecast": self._get_monthly_forecast_summary(db, municipality.dane_code),
            "catalog_crop_names": catalog_crop_names,
            "alternative_crops_from_ecocrop_fao": alternative_crops,
        }

    def get_or_generate(
        self,
        db: Session,
        municipality: Municipality,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Return an AI guide for the municipality."""
        now = datetime.now(timezone.utc)

        cached = (
            db.query(MunicipalityAIGuide)
            .filter(MunicipalityAIGuide.municipality_dane_code == municipality.dane_code)
            .first()
        )

        if cached and not force:
            if cached.expires_at and cached.expires_at > now:
                logger.info(
                    "[get_or_generate] Returning cached AI guide for municipality=%s (expires_at=%s)",
                    municipality.dane_code,
                    cached.expires_at,
                )
                return self._build_response(municipality, cached, cached=True)
            logger.info(
                "[get_or_generate] Cached AI guide expired for municipality=%s, regenerating",
                municipality.dane_code,
            )
        else:
            logger.info(
                "[get_or_generate] No cached AI guide for municipality=%s, generating",
                municipality.dane_code,
            )

        context = self._build_context(db, municipality)
        llm_result = self.llm_service.generate_municipality_ai_guide(context)

        tokens_in = llm_result.get("tokens_in")
        tokens_out = llm_result.get("tokens_out")
        tokens_total = None
        if tokens_in is not None and tokens_out is not None:
            tokens_total = tokens_in + tokens_out

        log_id = log_llm_generation(
            db,
            provider=llm_result.get("provider"),
            model=llm_result.get("model"),
            prompt_schema_version=PROMPT_SCHEMA_VERSION,
            context_summary=f"municipality_ai_guide:{municipality.dane_code}",
            response_json={
                "summary": llm_result.get("summary"),
                "alternative_crops": llm_result.get("alternative_crops"),
                "farming_systems": llm_result.get("farming_systems"),
                "soil_and_fertilizer": llm_result.get("soil_and_fertilizer"),
            },
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=llm_result.get("latency_ms"),
            status=llm_result.get("status", "llm_unavailable"),
            error_message=llm_result.get("error"),
        )
        logger.debug("[get_or_generate] LLM generation logged id=%s", log_id)

        if llm_result.get("status") != "success":
            return {
                "municipality_id": municipality.dane_code,
                "municipality_name": municipality.name,
                "summary": "",
                "alternative_crops": [],
                "farming_systems": [],
                "soil_and_fertilizer": [],
                "generated_at": None,
                "expires_at": None,
                "cached": False,
                "provider": llm_result.get("provider"),
                "model": llm_result.get("model"),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tokens_total": tokens_total,
                "latency_ms": llm_result.get("latency_ms"),
                "status": llm_result.get("status", "llm_unavailable"),
                "error": llm_result.get("error"),
            }

        content = json.dumps(
            {
                "summary": llm_result.get("summary"),
                "alternative_crops": llm_result.get("alternative_crops"),
                "farming_systems": llm_result.get("farming_systems"),
                "soil_and_fertilizer": llm_result.get("soil_and_fertilizer"),
            },
            ensure_ascii=False,
            default=str,
        )

        generated_at = now
        expires_at = now + timedelta(days=GUIDE_TTL_DAYS)

        if cached:
            cached.content = content
            cached.generated_at = generated_at
            cached.expires_at = expires_at
            cached.provider = llm_result.get("provider")
            cached.model = llm_result.get("model")
            cached.tokens_in = tokens_in
            cached.tokens_out = tokens_out
            cached.latency_ms = llm_result.get("latency_ms")
            cached.version = GUIDE_VERSION
        else:
            cached = MunicipalityAIGuide(
                municipality_dane_code=municipality.dane_code,
                content=content,
                generated_at=generated_at,
                expires_at=expires_at,
                provider=llm_result.get("provider"),
                model=llm_result.get("model"),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=llm_result.get("latency_ms"),
                version=GUIDE_VERSION,
            )
            db.add(cached)

        db.commit()
        db.refresh(cached)
        logger.info(
            "[get_or_generate] Saved AI guide for municipality=%s (expires_at=%s)",
            municipality.dane_code,
            expires_at,
        )

        return self._build_response(municipality, cached, cached=False)

    def _build_response(
        self,
        municipality: Municipality,
        guide: MunicipalityAIGuide,
        cached: bool,
    ) -> Dict[str, Any]:
        """Build the API response dict from a cached or freshly generated guide."""
        try:
            payload = json.loads(guide.content)
        except (json.JSONDecodeError, TypeError):
            payload = {
                "summary": "",
                "alternative_crops": [],
                "farming_systems": [],
                "soil_and_fertilizer": [],
            }

        return {
            "municipality_id": municipality.dane_code,
            "municipality_name": municipality.name,
            "summary": payload.get("summary", ""),
            "alternative_crops": payload.get("alternative_crops", []),
            "farming_systems": payload.get("farming_systems", []),
            "soil_and_fertilizer": payload.get("soil_and_fertilizer", []),
            "generated_at": guide.generated_at.isoformat() if guide.generated_at else None,
            "expires_at": guide.expires_at.isoformat() if guide.expires_at else None,
            "cached": cached,
            "provider": guide.provider,
            "model": guide.model,
            "tokens_in": guide.tokens_in,
            "tokens_out": guide.tokens_out,
            "tokens_total": (guide.tokens_in + guide.tokens_out) if guide.tokens_in and guide.tokens_out else None,
            "latency_ms": guide.latency_ms,
            "status": "success",
            "error": None,
        }


# Singleton
_municipality_ai_guide_service: Optional[MunicipalityAIGuideService] = None


def get_municipality_ai_guide_service() -> MunicipalityAIGuideService:
    global _municipality_ai_guide_service
    if _municipality_ai_guide_service is None:
        _municipality_ai_guide_service = MunicipalityAIGuideService()
    return _municipality_ai_guide_service
