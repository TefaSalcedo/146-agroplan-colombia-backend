"""Service for generating and caching national crop farmer guides.

Guides are generated with an LLM and cached for 3 months. If a fresh guide
exists in the database it is returned directly; otherwise a new one is
 generated and persisted.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.logger import get_logger
from app.models import Crop, CropNationalGuide
from app.services.llm_service import get_llm_service, log_llm_generation, PROMPT_SCHEMA_VERSION
from app.utils.crop_formatting import (
    format_duration_days,
    simplify_soil_terms,
    simplify_temperature,
    simplify_humidity,
    simplify_precipitation,
    simplify_altitude,
)

logger = get_logger("app.services.crop_national_guide_service")

GUIDE_VERSION = "1.0"
GUIDE_TTL_DAYS = 90


class CropNationalGuideService:
    """Manage LLM-generated national crop guides with 3-month caching."""

    def __init__(self):
        self.llm_service = get_llm_service()

    def _crop_to_dict(self, crop: Crop) -> Dict[str, Any]:
        """Serialize a Crop ORM object to a dict suitable for the LLM prompt."""
        return {
            "id": crop.id,
            "name": crop.name,
            "scientific_name": crop.scientific_name,
            "days_to_harvest": crop.days_to_harvest,
            "days_to_harvest_text": format_duration_days(crop.days_to_harvest),
            "establishment_period_days": crop.establishment_period_days,
            "establishment_period_text": format_duration_days(crop.establishment_period_days),
            "is_perennial": crop.is_perennial or False,
            "soil_type": crop.soil_type,
            "soil_type_simple": simplify_soil_terms(crop.soil_type or ""),
            "ideal_temperature": crop.ideal_temperature,
            "ideal_temperature_simple": simplify_temperature(crop.ideal_temperature or ""),
            "humidity": crop.humidity,
            "humidity_simple": simplify_humidity(crop.humidity or ""),
            "precipitation": crop.precipitation,
            "precipitation_simple": simplify_precipitation(crop.precipitation or ""),
            "altitude": crop.altitude,
            "altitude_simple": simplify_altitude(crop.altitude or ""),
            "irrigation": crop.irrigation,
            "substrates": crop.substrates or [],
            "planting_months": crop.planting_months or [],
            "harvest_months": crop.harvest_months or [],
            "tips": [
                {"title": tip.get("title", ""), "description": tip.get("description", "")}
                for tip in (crop.tips or [])
            ],
        }

    def get_or_generate(
        self,
        db: Session,
        crop: Crop,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Return a national farmer guide for the crop.

        Uses the cached guide if it exists and is still valid; otherwise
        generates a new one with the LLM, persists it and returns it.
        """
        now = datetime.now(timezone.utc)

        cached = (
            db.query(CropNationalGuide)
            .filter(CropNationalGuide.crop_id == crop.id)
            .first()
        )

        if cached and not force:
            if cached.expires_at and cached.expires_at > now:
                logger.info(
                    "[get_or_generate] Returning cached national guide for crop=%s (expires_at=%s)",
                    crop.id,
                    cached.expires_at,
                )
                return self._build_response(crop, cached, cached=True)
            logger.info(
                "[get_or_generate] Cached national guide expired for crop=%s, regenerating",
                crop.id,
            )
        else:
            logger.info(
                "[get_or_generate] No cached national guide for crop=%s, generating",
                crop.id,
            )

        crop_data = self._crop_to_dict(crop)
        llm_result = self.llm_service.generate_national_crop_guide(crop_data)

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
            context_summary=f"crop_national_guide:{crop.id}",
            response_json={
                "summary": llm_result.get("summary"),
                "sections": llm_result.get("sections"),
            },
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=llm_result.get("latency_ms"),
            status=llm_result.get("status", "llm_unavailable"),
            error_message=llm_result.get("error"),
        )
        logger.debug("[get_or_generate] LLM generation logged id=%s", log_id)

        if llm_result.get("status") != "success":
            logger.warning(
                "[get_or_generate] LLM failed for crop=%s: %s",
                crop.id,
                llm_result.get("error"),
            )
            return {
                "crop_id": crop.id,
                "crop_name": crop.name,
                "summary": "",
                "sections": [],
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
                "sections": llm_result.get("sections"),
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
            cached = CropNationalGuide(
                crop_id=crop.id,
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
            "[get_or_generate] Saved national guide for crop=%s (expires_at=%s)",
            crop.id,
            expires_at,
        )

        return self._build_response(crop, cached, cached=False)

    def _build_response(
        self,
        crop: Crop,
        guide: CropNationalGuide,
        cached: bool,
    ) -> Dict[str, Any]:
        """Build the API response dict from a cached or freshly generated guide."""
        try:
            payload = json.loads(guide.content)
        except (json.JSONDecodeError, TypeError):
            payload = {"summary": "", "sections": []}

        sections = payload.get("sections", [])
        if not isinstance(sections, list):
            sections = []

        return {
            "crop_id": crop.id,
            "crop_name": crop.name,
            "summary": payload.get("summary", ""),
            "sections": sections,
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
_crop_national_guide_service: Optional[CropNationalGuideService] = None


def get_crop_national_guide_service() -> CropNationalGuideService:
    global _crop_national_guide_service
    if _crop_national_guide_service is None:
        _crop_national_guide_service = CropNationalGuideService()
    return _crop_national_guide_service
