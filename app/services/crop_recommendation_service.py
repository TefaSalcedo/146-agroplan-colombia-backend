"""Service for generating and caching crop-municipality recommendations.

Recommendations are generated with an LLM and cached for 5 days. If a fresh
recommendation exists in the database it is returned directly; otherwise a new
one is generated and persisted.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.logger import get_logger
from app.models import Crop, CropMunicipalityRecommendation, Municipality
from app.services.llm_service import get_llm_service, log_llm_generation, PROMPT_SCHEMA_VERSION
from app.services.prediction_service import PredictionService

logger = get_logger("app.services.crop_recommendation_service")

CACHE_TTL_DAYS = 5
RECOMMENDATION_VERSION = "1.0"


class CropRecommendationCacheService:
    """Manage LLM-generated crop-municipality recommendations with 5-day cache."""

    def __init__(self):
        self.llm_service = get_llm_service()
        self.prediction_service = PredictionService()

    def get_or_generate(
        self,
        db: Session,
        crop: Crop,
        municipality: Municipality,
    ) -> Dict[str, Any]:
        """Return a cached or freshly generated recommendation.

        When LLM is disabled, return an empty response without reading or
        writing the recommendation cache.

        Uses an advisory lock to prevent concurrent requests from generating
        duplicate recommendations for the same crop-municipality pair.
        """
        now = datetime.now(timezone.utc)
        dane = municipality.dane_code
        crop_id = crop.id

        if not get_settings().llm_enabled:
            return {
                "crop_id": crop_id,
                "crop_name": crop.name,
                "municipality_id": dane,
                "municipality_name": municipality.name,
                "text": "",
                "cached": False,
                "generated_at": None,
                "expires_at": None,
                "provider": None,
                "model": None,
                "tokens_in": None,
                "tokens_out": None,
                "tokens_total": None,
                "latency_ms": None,
                "status": "llm_disabled",
                "error": None,
            }

        def _fetch_recommendation() -> Optional[CropMunicipalityRecommendation]:
            return (
                db.query(CropMunicipalityRecommendation)
                .filter(
                    CropMunicipalityRecommendation.crop_id == crop_id,
                    CropMunicipalityRecommendation.municipality_dane_code == dane,
                )
                .first()
            )

        cached = _fetch_recommendation()

        if cached and cached.expires_at and cached.expires_at > now:
            logger.info(
                "[get_or_generate] Returning cached recommendation crop=%s municipality=%s (expires_at=%s)",
                crop_id,
                dane,
                cached.expires_at,
            )
            return self._build_response(crop, municipality, cached, cached=True)

        if cached:
            logger.info(
                "[get_or_generate] Cached recommendation expired for crop=%s municipality=%s, regenerating",
                crop_id,
                dane,
            )
        else:
            logger.info(
                "[get_or_generate] No cached recommendation for crop=%s municipality=%s, generating",
                crop_id,
                dane,
            )

        # Advisory lock scoped to this crop-municipality pair.
        lock_key = abs(hash(f"crop_recommendation:{crop_id}:{dane}")) % (2**31)
        logger.debug("[get_or_generate] Acquiring advisory lock key=%s", lock_key)
        db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock_key})

        # Double-check after acquiring the lock.
        cached = _fetch_recommendation()
        if cached and cached.expires_at and cached.expires_at > now:
            logger.info(
                "[get_or_generate] Cache hit after lock crop=%s municipality=%s (expires_at=%s)",
                crop_id,
                dane,
                cached.expires_at,
            )
            return self._build_response(crop, municipality, cached, cached=True)

        context = self.prediction_service.get_crop_recommendation_context(
            db,
            crop_id=crop_id,
            municipality_id=dane,
        )
        if not context:
            logger.error("[get_or_generate] Could not build recommendation context")
            return {
                "crop_id": crop_id,
                "crop_name": crop.name,
                "municipality_id": dane,
                "municipality_name": municipality.name,
                "text": "",
                "cached": False,
                "generated_at": None,
                "expires_at": None,
                "provider": None,
                "model": None,
                "tokens_in": None,
                "tokens_out": None,
                "tokens_total": None,
                "latency_ms": None,
                "status": "context_unavailable",
                "error": "Could not build recommendation context",
            }

        llm_result = self.llm_service.generate_crop_recommendation(
            crop=context["crop"],
            municipality=context["municipality"],
            climate_summary=context["climate_summary"],
            soil_summary=context["soil_summary"],
        )

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
            context_summary=f"crop_recommendation:{crop_id}:{dane}",
            response_json={"text": llm_result.get("text")},
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=llm_result.get("latency_ms"),
            status=llm_result.get("status", "llm_unavailable"),
            error_message=llm_result.get("error"),
        )
        logger.debug("[get_or_generate] LLM generation logged id=%s", log_id)

        if llm_result.get("status") != "success":
            return {
                "crop_id": crop_id,
                "crop_name": crop.name,
                "municipality_id": dane,
                "municipality_name": municipality.name,
                "text": "",
                "cached": False,
                "generated_at": None,
                "expires_at": None,
                "provider": llm_result.get("provider"),
                "model": llm_result.get("model"),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tokens_total": tokens_total,
                "latency_ms": llm_result.get("latency_ms"),
                "status": llm_result.get("status", "llm_unavailable"),
                "error": llm_result.get("error"),
            }

        generated_at = now
        expires_at = now + timedelta(days=CACHE_TTL_DAYS)

        if cached:
            cached.text = llm_result.get("text", "")
            cached.generated_at = generated_at
            cached.expires_at = expires_at
            cached.provider = llm_result.get("provider")
            cached.model = llm_result.get("model")
            cached.tokens_in = tokens_in
            cached.tokens_out = tokens_out
            cached.latency_ms = llm_result.get("latency_ms")
            cached.version = RECOMMENDATION_VERSION
        else:
            cached = CropMunicipalityRecommendation(
                crop_id=crop_id,
                municipality_dane_code=dane,
                text=llm_result.get("text", ""),
                generated_at=generated_at,
                expires_at=expires_at,
                provider=llm_result.get("provider"),
                model=llm_result.get("model"),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=llm_result.get("latency_ms"),
                version=RECOMMENDATION_VERSION,
            )
            db.add(cached)

        try:
            db.commit()
            db.refresh(cached)
        except IntegrityError:
            db.rollback()
            logger.warning(
                "[get_or_generate] Race condition detected crop=%s municipality=%s, fetching existing recommendation",
                crop_id,
                dane,
            )
            cached = _fetch_recommendation()
            if cached is None:
                logger.error("[get_or_generate] Could not fetch existing recommendation crop=%s municipality=%s after race", crop_id, dane)
                return {
                    "crop_id": crop_id,
                    "crop_name": crop.name,
                    "municipality_id": dane,
                    "municipality_name": municipality.name,
                    "text": llm_result.get("text", ""),
                    "cached": False,
                    "generated_at": None,
                    "expires_at": None,
                    "provider": llm_result.get("provider"),
                    "model": llm_result.get("model"),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "tokens_total": tokens_total,
                    "latency_ms": llm_result.get("latency_ms"),
                    "status": "success",
                    "error": None,
                }
            return self._build_response(crop, municipality, cached, cached=True)

        logger.info(
            "[get_or_generate] Saved recommendation crop=%s municipality=%s (expires_at=%s)",
            crop_id,
            dane,
            expires_at,
        )

        return self._build_response(crop, municipality, cached, cached=False)

    def _build_response(
        self,
        crop: Crop,
        municipality: Municipality,
        recommendation: CropMunicipalityRecommendation,
        cached: bool,
    ) -> Dict[str, Any]:
        tokens_total = None
        if recommendation.tokens_in is not None and recommendation.tokens_out is not None:
            tokens_total = recommendation.tokens_in + recommendation.tokens_out

        return {
            "crop_id": crop.id,
            "crop_name": crop.name,
            "municipality_id": municipality.dane_code,
            "municipality_name": municipality.name,
            "text": recommendation.text,
            "cached": cached,
            "generated_at": recommendation.generated_at.isoformat() if recommendation.generated_at else None,
            "expires_at": recommendation.expires_at.isoformat() if recommendation.expires_at else None,
            "provider": recommendation.provider,
            "model": recommendation.model,
            "tokens_in": recommendation.tokens_in,
            "tokens_out": recommendation.tokens_out,
            "tokens_total": tokens_total,
            "latency_ms": recommendation.latency_ms,
            "status": "success",
            "error": None,
        }


# Singleton
_crop_recommendation_cache_service: Optional[CropRecommendationCacheService] = None


def get_crop_recommendation_cache_service() -> CropRecommendationCacheService:
    global _crop_recommendation_cache_service
    if _crop_recommendation_cache_service is None:
        _crop_recommendation_cache_service = CropRecommendationCacheService()
    return _crop_recommendation_cache_service
