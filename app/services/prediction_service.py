"""Prediction service that orchestrates ML models, fallbacks, cache and audit.

When ML models are loaded (via ModelLoader), this service uses them for inference.
When models are not available, it falls back to the MockPredictor for development.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import get_settings
from app.models import PredictionCache, PredictionRun
from app.services.mock_predictor import MockPredictor

settings = get_settings()


def _build_yield_features(db: Session, crop_id: str, municipality_id: str) -> Optional[Any]:
    """Build the feature vector expected by the yield ensemble.

    This is a placeholder that will be fully implemented once the yield
    preprocessor and municipality/yield profiles are available. It currently
    returns None so the service falls back to mock predictions.
    """
    # TODO: implement when yield_profiles.parquet and preprocessor.pkl are ready
    return None


def _predict_yield_with_ensemble(db: Session, crop_id: str, municipality_id: str) -> Optional[Dict]:
    """Run the XGBoost 0.65 / LightGBM 0.35 yield ensemble if artifacts are ready.

    Returns None if any required artifact is missing so the caller can fall
    back to mock predictions.
    """
    try:
        from app.services.model_loader import get_model_loader

        loader = get_model_loader()
        if not loader.is_yield_model_loaded():
            return None

        X = _build_yield_features(db, crop_id, municipality_id)
        if X is None:
            return None

        prediction = loader.predict_yield_ensemble(X)
        if prediction is None:
            return None

        return {
            "yield_prediction": prediction,
            "yield_model_version": "yield-ensemble-v1",
            "yield_confidence": "medium" if loader.is_yield_model_loaded() else "low",
            "method": "yield_ensemble",
        }
    except Exception:
        return None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _compute_cache_expiry() -> datetime:
    """Compute cache expiry as end of current month in America/Bogota timezone."""
    import pytz

    bogota_tz = pytz.timezone("America/Bogota")
    now_bogota = datetime.now(bogota_tz)

    # End of current month: first day of next month at 00:00 Bogota
    if now_bogota.month == 12:
        first_next = now_bogota.replace(
            year=now_bogota.year + 1, month=1, day=1,
            hour=0, minute=0, second=0, microsecond=0,
        )
    else:
        first_next = now_bogota.replace(
            month=now_bogota.month + 1, day=1,
            hour=0, minute=0, second=0, microsecond=0,
        )

    return first_next.astimezone(timezone.utc)


def _make_cache_key(prediction_type: str, **kwargs) -> str:
    """Build a deterministic cache key from prediction type and parameters."""
    raw = json.dumps(kwargs, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"{prediction_type}:{digest}"


class PredictionService:
    """Orchestrates predictions with cache, model selection, and audit logging."""

    def __init__(self):
        self._mock = MockPredictor()
        self._model_loader = None

    def _get_model_loader(self):
        """Lazily get the model loader singleton."""
        if self._model_loader is None:
            try:
                from app.services.model_loader import get_model_loader

                self._model_loader = get_model_loader()
            except Exception:
                self._model_loader = None
        return self._model_loader

    def _check_cache(self, db: Session, cache_key: str) -> Optional[dict]:
        """Check if a valid cache entry exists."""
        entry = (
            db.query(PredictionCache)
            .filter(PredictionCache.cache_key == cache_key)
            .filter(PredictionCache.expires_at > _utcnow())
            .first()
        )
        if entry:
            return entry.payload
        return None

    def _store_cache(
        self,
        db: Session,
        cache_key: str,
        prediction_type: str,
        payload: dict,
        scope_key: Optional[str] = None,
    ) -> None:
        """Store a prediction in the cache."""
        expiry = _compute_cache_expiry()
        entry = PredictionCache(
            cache_key=cache_key,
            prediction_type=prediction_type,
            scope_key=scope_key,
            payload=payload,
            expires_at=expiry,
        )
        db.merge(entry)
        db.commit()

    def _log_run(
        self,
        db: Session,
        prediction_type: str,
        inputs: dict,
        result: dict,
        cache_hit: bool,
        latency_ms: int,
        method: str,
        fallback_used: bool = False,
        missing_features: Optional[List[str]] = None,
        status: str = "success",
        error_message: Optional[str] = None,
    ) -> None:
        """Log a prediction run for audit."""
        run = PredictionRun(
            prediction_type=prediction_type,
            inputs=inputs,
            result=result,
            cache_hit=cache_hit,
            models_used=None,
            missing_features=missing_features,
            method=method,
            fallback_used=fallback_used,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
        )
        db.add(run)
        db.commit()

    def predict_zoning(
        self,
        db: Session,
        crop_id: str,
        municipality_id: str,
    ) -> Dict:
        """Predict zoning suitability for a crop in a municipality.

        Flow:
        1. Check cache
        2. Acquire advisory lock
        3. Double-check cache
        4. Run inference (primary model or fallback)
        5. Store in cache
        6. Log audit
        """
        cache_key = _make_cache_key("zoning", crop_id=crop_id, municipality_id=municipality_id)

        # Step 1: Check cache
        cached = self._check_cache(db, cache_key)
        if cached:
            cached["cache_hit"] = True
            return cached

        # Step 2: Advisory lock
        lock_key = abs(hash(cache_key)) % (2**31)
        db.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": lock_key})

        # Step 3: Double-check cache
        cached = self._check_cache(db, cache_key)
        if cached:
            cached["cache_hit"] = True
            return cached

        # Step 4: Inference
        start = time.time()
        loader = self._get_model_loader()

        if loader and loader.is_zoning_model_loaded():
            # Real ML inference path (future)
            result = self._mock.predict_zoning(db, crop_id, municipality_id)
            method = "primary_model"
            fallback_used = False
        else:
            # Mock fallback for development
            result = self._mock.predict_zoning(db, crop_id, municipality_id)
            method = "mock"
            fallback_used = False

        latency_ms = int((time.time() - start) * 1000)

        # Enrich result with method info
        result["method"] = method
        result["cache_hit"] = False

        # Step 5: Store cache
        self._store_cache(db, cache_key, "zoning", result, scope_key=f"{crop_id}:{municipality_id}")

        # Step 6: Log audit
        self._log_run(
            db=db,
            prediction_type="zoning",
            inputs={"crop_id": crop_id, "municipality_id": municipality_id},
            result=result,
            cache_hit=False,
            latency_ms=latency_ms,
            method=method,
            fallback_used=fallback_used,
        )

        return result

    def predict_calendar_batch(
        self,
        db: Session,
        municipality_id: str,
        crop_ids: List[str],
        horizon_months: int = 12,
    ) -> Dict:
        """Predict planting calendars for multiple crops in a municipality.

        This is a placeholder that returns mock data. When yield models are
        loaded, it will use the XGBoost/LightGBM ensemble.
        """
        cache_key = _make_cache_key(
            "calendar_batch",
            municipality_id=municipality_id,
            crop_ids=sorted(crop_ids),
            horizon_months=horizon_months,
        )

        cached = self._check_cache(db, cache_key)
        if cached:
            return cached

        start = time.time()
        results = []

        for crop_id in crop_ids:
            result = self._build_calendar_crop_result(db, crop_id, municipality_id, horizon_months)
            results.append(result)

        latency_ms = int((time.time() - start) * 1000)
        payload = {
            "municipality_id": municipality_id,
            "horizon_months": horizon_months,
            "results": results,
            "model_version": "mock-v1",
            "explanation": None,
            "llm_status": "llm_unavailable",
        }

        self._store_cache(
            db, cache_key, "calendar_batch", payload,
            scope_key=f"{municipality_id}:{','.join(sorted(crop_ids))}",
        )

        self._log_run(
            db=db,
            prediction_type="calendar_batch",
            inputs={
                "municipality_id": municipality_id,
                "crop_ids": crop_ids,
                "horizon_months": horizon_months,
            },
            result=payload,
            cache_hit=False,
            latency_ms=latency_ms,
            method="mock",
        )

        return payload

    def _build_calendar_crop_result(
        self,
        db: Session,
        crop_id: str,
        municipality_id: str,
        horizon_months: int,
    ) -> Dict:
        """Build a mock calendar result for a single crop."""
        from app.services.crop_catalog import CropCatalog
        from app.services.municipality_catalog import MunicipalityCatalog
        from app.models import Municipality, MunicipalityClimateForecast
        from datetime import timedelta

        crop_catalog = CropCatalog()
        municipality_catalog = MunicipalityCatalog()
        crop = crop_catalog.get_crop_model_by_id(db, crop_id)
        municipality = municipality_catalog.get_municipality_by_id(db, municipality_id)
        if not crop or not municipality:
            return {
                "crop_id": crop_id,
                "crop_name": crop_id,
                "yield_prediction": None,
                "yield_model_version": None,
                "yield_confidence": "low",
                "top_harvest_months": [],
                "monthly_forecasts": [],
                "warnings": ["Crop not found"],
                "method": "mock",
            }

        # Build monthly forecasts from stored data or mock
        today = _utcnow().date()
        monthly_forecasts = []

        for i in range(horizon_months):
            month = ((today.month + i - 1) % 12) + 1
            year = today.year + ((today.month + i - 1) // 12)

            # Try to get stored forecast data
            records = (
                db.query(MunicipalityClimateForecast)
                .filter(MunicipalityClimateForecast.municipality_dane_code == municipality_id)
                .filter(MunicipalityClimateForecast.forecast_date >= today.replace(day=1) + timedelta(days=30 * i))
                .filter(MunicipalityClimateForecast.forecast_date < today.replace(day=1) + timedelta(days=30 * (i + 1)))
                .all()
            )

            if records:
                avg_temp = sum(r.temp_mean for r in records if r.temp_mean) / max(1, len(records))
                avg_precip = sum(r.precipitation for r in records if r.precipitation is not None) / max(1, len(records))
                avg_humidity = sum(r.humidity for r in records if r.humidity) / max(1, len(records))
                climate_source = "open_meteo_forecast"
            else:
                avg_temp = 22.0
                avg_precip = 100.0
                avg_humidity = 75.0
                climate_source = "historical_climatology"

            monthly_forecasts.append({
                "month": month,
                "year": year,
                "temp_mean": round(avg_temp, 1) if avg_temp else None,
                "precipitation": round(avg_precip, 1) if avg_precip else None,
                "humidity": round(avg_humidity, 1) if avg_humidity else None,
                "climate_source": climate_source,
            })

        MONTHS_LONG = [
            "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
            "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
        ]

        # Try real yield ensemble first; fall back to mock if artifacts are not ready
        yield_result = _predict_yield_with_ensemble(db, crop_id, municipality.dane_code)

        # Mock top 3 harvest months (future: use yield model + EVA/FAO)
        top_harvest = []
        for i in range(min(3, horizon_months)):
            mf = monthly_forecasts[i]
            top_harvest.append({
                "harvest_month": mf["month"],
                "harvest_year": mf["year"],
                "harvest_month_name": MONTHS_LONG[mf["month"] - 1],
                "score": 0.75 - (i * 0.1),
                "planting_months": crop.planting_months or [],
                "planting_year": mf["year"],
                "duration_days_min": crop.days_to_harvest,
                "duration_days_max": crop.days_to_harvest,
            })

        if yield_result:
            return {
                "crop_id": crop_id,
                "crop_name": crop.name,
                "yield_prediction": yield_result["yield_prediction"],
                "yield_model_version": yield_result["yield_model_version"],
                "yield_confidence": yield_result["yield_confidence"],
                "top_harvest_months": top_harvest,
                "monthly_forecasts": monthly_forecasts,
                "warnings": [],
                "method": yield_result["method"],
            }

        return {
            "crop_id": crop_id,
            "crop_name": crop.name,
            "yield_prediction": None,
            "yield_model_version": None,
            "yield_confidence": "low",
            "top_harvest_months": top_harvest,
            "monthly_forecasts": monthly_forecasts,
            "warnings": ["Mock prediction - ML models not yet loaded"],
            "method": "mock",
        }
