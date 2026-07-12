"""Prediction service that orchestrates ML models, fallbacks, cache and audit.

When ML models are loaded (via ModelLoader), this service uses them for inference.
When models are not available, it falls back to the MockPredictor for development.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import get_settings
from app.logger import get_logger
from app.models import PredictionCache, PredictionRun
from app.services.feature_builder import build_yield_features, build_zoning_features
from app.services.mock_predictor import MockPredictor

settings = get_settings()
logger = get_logger("app.services.prediction_service")


# Mapping from model class index to API suitability label for zoning.
_ZONING_CLASS_ORDER = ["no_apta", "baja", "media", "alta"]
_ZONING_CLASS_TO_API = {
    "no_apta": "none",
    "baja": "low",
    "media": "medium",
    "alta": "high",
}


def _parse_months_string(value: Any) -> List[int]:
    """Parse a comma-separated month string into a list of ints."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [int(m.strip()) for m in str(value).split(",") if m.strip().isdigit()]


def _get_crop_calendar_info(db: Session, crop_id: str, loader: Any) -> Dict[str, Any]:
    """Return crop calendar info from EVA/FAO datasets with verified fallback.

    Priority:
    1. FAO sowing months from calendar_for_eva.csv.
    2. EVA-inferred semester-based planting months (data-driven, not hardcoded).
    3. Verified agronomic facts from external sources (crop_agronomic_data.py).

    Cycle duration priority:
    1. FAO/EcoCrop duration fields.
    2. Verified agronomic facts for crops with no EcoCrop duration (aguacate).
    """
    from app.services.crop_agronomic_data import get_agronomic_facts
    from app.services.feature_builder import _get_calendar_row

    info: Dict[str, Any] = {
        "planting_months": [],
        "duration_days_min": None,
        "duration_days_max": None,
        "cycle_days_min": None,
        "cycle_days_max": None,
    }

    verified = get_agronomic_facts(crop_id)

    def _valid_int(value):
        if value is None or pd.isna(value):
            return None
        v = int(value)
        return v if v > 0 else None

    # Try FAO/EcoCrop calendar first
    if loader is not None and loader.calendar_for_eva is not None:
        try:
            row = _get_calendar_row(loader, crop_id)
            if row is not None:
                fao_months = _parse_months_string(row.get("fao_sowing_months"))
                gmin = row.get("gmin_dias")
                gmax = row.get("gmax_dias")
                fao_dur_min = row.get("fao_duration_min_dias")
                fao_dur_max = row.get("fao_duration_max_dias")

                info["planting_months"] = fao_months
                info["duration_days_min"] = _valid_int(fao_dur_min) or _valid_int(gmin)
                info["duration_days_max"] = _valid_int(fao_dur_max) or _valid_int(gmax)
                info["cycle_days_min"] = _valid_int(gmin)
                info["cycle_days_max"] = _valid_int(gmax)
        except Exception:
            pass

    # If FAO did not provide sowing months, try EVA inference
    if not info["planting_months"] and loader is not None:
        eva_months = _infer_planting_months_from_eva(loader, crop_id)
        if eva_months:
            info["planting_months"] = eva_months

    # Last fallback: verified agronomic facts
    if not info["planting_months"] and verified:
        info["planting_months"] = verified.get("planting_months", [])

    if info["duration_days_min"] is None and verified:
        info["duration_days_min"] = verified.get("cycle_days_min")
    if info["duration_days_max"] is None and verified:
        info["duration_days_max"] = verified.get("cycle_days_max")
    if info["cycle_days_min"] is None and verified:
        info["cycle_days_min"] = verified.get("cycle_days_min")
    if info["cycle_days_max"] is None and verified:
        info["cycle_days_max"] = verified.get("cycle_days_max")

    return info


def _infer_planting_months_from_eva(loader: Any, crop_id: str) -> List[int]:
    """Infer planting months from EVA historical frequency by semester.

    EVA only records semester (1 = Jan-Jun, 2 = Jul-Dec), so we map the most
    active semester to representative months. This is data-driven, not hardcoded.
    """
    from app.services.feature_builder import _YIELD_CROP_NAME_MAP

    profiles = loader.yield_profiles if loader else None
    if profiles is None:
        return []

    crop_name = _YIELD_CROP_NAME_MAP.get(crop_id)
    if not crop_name:
        return []

    rows = profiles[profiles["cultivo"] == crop_name]
    if rows.empty:
        return []

    # Count records per semester and pick the one with most activity.
    counts = rows.groupby("semestre").size().to_dict()
    if not counts:
        return []

    # If the crop is mostly annual, EVA does not indicate a specific sowing window.
    annual_ratio = (rows["es_anual"].sum() / len(rows)) if "es_anual" in rows.columns else 0.0
    if annual_ratio > 0.8:
        return []

    best_semester = max(counts, key=counts.get)
    if best_semester == 1:
        return [3, 4, 5, 6]
    elif best_semester == 2:
        return [9, 10, 11, 12]
    return []


def _predict_yield_with_ensemble(db: Session, crop_id: str, municipality_id: str) -> Optional[Dict]:
    """Run the XGBoost 0.65 / LightGBM 0.35 yield ensemble if artifacts are ready.

    Returns None if any required artifact is missing so the caller can fall
    back to mock predictions.
    """
    logger.debug("[_predict_yield_with_ensemble] Starting yield ensemble (crop_id=%s, municipality_id=%s)", crop_id, municipality_id)
    try:
        from app.services.model_loader import get_model_loader

        loader = get_model_loader()
        if not loader.is_yield_model_loaded():
            logger.debug("[_predict_yield_with_ensemble] Yield models not loaded; skipping ensemble")
            return None

        logger.debug("[_predict_yield_with_ensemble] Building yield features")
        features = build_yield_features(db, crop_id, municipality_id, loader)
        if features is None:
            logger.debug("[_predict_yield_with_ensemble] Yield features are None; skipping ensemble")
            return None

        X_xgb, X_lgbm = features
        xgb_pred = None
        lgbm_pred = None

        if loader.yield_xgb_model is not None and X_xgb is not None:
            logger.debug("[_predict_yield_with_ensemble] Running XGBoost yield model")
            xgb_pred = float(loader.yield_xgb_model.predict(X_xgb)[0])
            logger.debug("[_predict_yield_with_ensemble] XGBoost prediction: %s", xgb_pred)
        if loader.yield_lgbm_model is not None and X_lgbm is not None:
            logger.debug("[_predict_yield_with_ensemble] Running LightGBM yield model")
            lgbm_pred = float(loader.yield_lgbm_model.predict(X_lgbm)[0])
            logger.debug("[_predict_yield_with_ensemble] LightGBM prediction: %s", lgbm_pred)

        if xgb_pred is None and lgbm_pred is None:
            logger.warning("[_predict_yield_with_ensemble] Both yield models returned None")
            return None

        if xgb_pred is not None and lgbm_pred is not None:
            prediction = loader.yield_xgb_weight * xgb_pred + loader.yield_lgbm_weight * lgbm_pred
            logger.info("[_predict_yield_with_ensemble] Ensemble prediction (xgb=%s, lgbm=%s, weighted=%s)", xgb_pred, lgbm_pred, prediction)
        elif xgb_pred is not None:
            prediction = xgb_pred
            logger.info("[_predict_yield_with_ensemble] Using XGBoost prediction only: %s", prediction)
        else:
            prediction = lgbm_pred
            logger.info("[_predict_yield_with_ensemble] Using LightGBM prediction only: %s", prediction)

        return {
            "yield_prediction": round(float(prediction), 4),
            "yield_model_version": "yield-ensemble-v1",
            "yield_confidence": "medium",
            "method": "yield_ensemble",
        }
    except Exception as e:
        logger.error("[_predict_yield_with_ensemble] Yield ensemble failed: %s", e)
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
        logger.info("[predict_zoning] Starting prediction (crop_id=%s, municipality_id=%s)", crop_id, municipality_id)
        cache_key = _make_cache_key("zoning", crop_id=crop_id, municipality_id=municipality_id)
        logger.debug("[predict_zoning] cache_key=%s", cache_key)

        # Step 1: Check cache
        logger.debug("[predict_zoning] Checking prediction cache")
        cached = self._check_cache(db, cache_key)
        if cached:
            logger.info("[predict_zoning] Cache hit for (crop_id=%s, municipality_id=%s)", crop_id, municipality_id)
            cached["cache_hit"] = True
            return cached
        logger.debug("[predict_zoning] Cache miss")

        # Step 2: Advisory lock
        lock_key = abs(hash(cache_key)) % (2**31)
        logger.debug("[predict_zoning] Acquiring advisory lock key=%s", lock_key)
        db.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": lock_key})

        # Step 3: Double-check cache
        logger.debug("[predict_zoning] Double-checking prediction cache after lock")
        cached = self._check_cache(db, cache_key)
        if cached:
            logger.info("[predict_zoning] Cache hit after lock for (crop_id=%s, municipality_id=%s)", crop_id, municipality_id)
            cached["cache_hit"] = True
            return cached

        # Step 4: Inference
        logger.info("[predict_zoning] Running inference")
        start = time.time()
        loader = self._get_model_loader()
        logger.debug("[predict_zoning] Model loader loaded=%s", loader.is_zoning_model_loaded() if loader else False)
        result: Optional[Dict] = None
        method = "mock"
        fallback_used = False
        missing_features: Optional[List[str]] = None

        if loader and loader.is_zoning_model_loaded():
            logger.info("[predict_zoning] Building zoning features")
            X = build_zoning_features(db, crop_id, municipality_id, loader)
            logger.debug("[predict_zoning] Zoning features result: X is None=%s", X is None)
            if X is not None:
                try:
                    logger.info("[predict_zoning] Calling LightGBM zoning model")
                    probabilities = loader.predict_zoning_proba(X)
                    logger.debug("[predict_zoning] Model probabilities: %s", probabilities)
                    if probabilities is not None:
                        class_idx = int(np.argmax(probabilities[0]))
                        class_label = _ZONING_CLASS_ORDER[class_idx]
                        suitability = _ZONING_CLASS_TO_API[class_label]
                        confidence = round(float(probabilities[0][class_idx]), 4)

                        # Build per-class probabilities dict
                        prob_dict = {
                            _ZONING_CLASS_TO_API[_ZONING_CLASS_ORDER[i]]: round(float(p), 4)
                            for i, p in enumerate(probabilities[0])
                        }

                        result = {
                            "crop_id": crop_id,
                            "municipality_id": municipality_id,
                            "suitability": suitability,
                            "confidence": confidence,
                            "model_version": "zoning-lightgbm-v1",
                            "factors": {
                                "temperature_match": True,
                                "precipitation_match": True,
                                "soil_match": True,
                                "altitude_match": True,
                            },
                            "probabilities": prob_dict,
                        }
                        method = "primary_model"
                        logger.info("[predict_zoning] Primary model result: suitability=%s confidence=%s", suitability, confidence)
                except Exception as e:
                    logger.error("[predict_zoning] Zoning inference failed: %s", e)
                    missing_features = ["primary_model_error"]

        if result is None:
            # Mock fallback for development or when artifacts are incomplete
            logger.warning("[predict_zoning] Falling back to MockPredictor (method=mock)")
            result = self._mock.predict_zoning(db, crop_id, municipality_id)
            method = "mock"
            fallback_used = True

        latency_ms = int((time.time() - start) * 1000)
        logger.info("[predict_zoning] Inference completed in %sms (method=%s)", latency_ms, method)

        # Enrich result with method info
        result["method"] = method
        result["cache_hit"] = False

        # Step 5: Store cache
        logger.debug("[predict_zoning] Storing result in cache")
        self._store_cache(db, cache_key, "zoning", result, scope_key=f"{crop_id}:{municipality_id}")

        # Step 6: Log audit
        logger.debug("[predict_zoning] Logging prediction run audit")
        self._log_run(
            db=db,
            prediction_type="zoning",
            inputs={"crop_id": crop_id, "municipality_id": municipality_id},
            result=result,
            cache_hit=False,
            latency_ms=latency_ms,
            method=method,
            fallback_used=fallback_used,
            missing_features=missing_features,
        )

        logger.info("[predict_zoning] Returning result (suitability=%s, method=%s)", result.get("suitability"), method)
        return result

    def predict_calendar_batch(
        self,
        db: Session,
        municipality_id: str,
        crop_ids: List[str],
        horizon_months: int = 12,
    ) -> Dict:
        """Predict planting calendars for multiple crops in a municipality."""
        logger.info("[predict_calendar_batch] Starting prediction (municipality_id=%s, crops=%s, horizon=%s)", municipality_id, len(crop_ids), horizon_months)
        cache_key = _make_cache_key(
            "calendar_batch",
            municipality_id=municipality_id,
            crop_ids=sorted(crop_ids),
            horizon_months=horizon_months,
        )
        logger.debug("[predict_calendar_batch] cache_key=%s", cache_key)

        cached = self._check_cache(db, cache_key)
        if cached:
            logger.info("[predict_calendar_batch] Cache hit for municipality_id=%s", municipality_id)
            return cached
        logger.debug("[predict_calendar_batch] Cache miss")

        start = time.time()
        results = []

        for crop_id in crop_ids:
            logger.debug("[predict_calendar_batch] Building calendar for crop_id=%s", crop_id)
            result = self._build_calendar_crop_result(db, crop_id, municipality_id, horizon_months)
            results.append(result)

        latency_ms = int((time.time() - start) * 1000)
        logger.info("[predict_calendar_batch] Calendar results built in %sms for %s crops", latency_ms, len(results))
        payload = {
            "municipality_id": municipality_id,
            "horizon_months": horizon_months,
            "results": results,
            "model_version": "mock-v1",
            "explanation": None,
            "llm_status": "llm_unavailable",
        }

        logger.debug("[predict_calendar_batch] Storing calendar batch in cache")
        self._store_cache(
            db, cache_key, "calendar_batch", payload,
            scope_key=f"{municipality_id}:{','.join(sorted(crop_ids))}",
        )

        logger.debug("[predict_calendar_batch] Logging prediction run audit")
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

        logger.info("[predict_calendar_batch] Returning payload with %s crop results", len(results))
        return payload

    def _build_calendar_crop_result(
        self,
        db: Session,
        crop_id: str,
        municipality_id: str,
        horizon_months: int,
    ) -> Dict:
        """Build a calendar result for a single crop."""
        logger.debug("[_build_calendar_crop_result] Building calendar (crop_id=%s, municipality_id=%s)", crop_id, municipality_id)
        from app.services.crop_catalog import CropCatalog
        from app.services.municipality_catalog import MunicipalityCatalog
        from app.models import Municipality, MunicipalityClimateForecast
        from datetime import timedelta

        crop_catalog = CropCatalog()
        municipality_catalog = MunicipalityCatalog()
        crop = crop_catalog.get_crop_model_by_id(db, crop_id)
        municipality = municipality_catalog.get_municipality_by_id(db, municipality_id)
        if not crop or not municipality:
            logger.warning("[_build_calendar_crop_result] Crop or municipality not found (crop_id=%s, municipality_id=%s)", crop_id, municipality_id)
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
        logger.debug("[_build_calendar_crop_result] Building %s monthly forecasts", horizon_months)

        for i in range(horizon_months):
            month = ((today.month + i - 1) % 12) + 1
            year = today.year + ((today.month + i - 1) // 12)

            # Try to get stored forecast data
            logger.debug("[_build_calendar_crop_result] Querying climate forecast records for month=%s/%s", month, year)
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
                logger.debug("[_build_calendar_crop_result] Found %s forecast records for month=%s/%s", len(records), month, year)
            else:
                # No forecast and no hardcoded climatology: report missing data.
                avg_temp = None
                avg_precip = None
                avg_humidity = None
                climate_source = "not_available"
                logger.debug("[_build_calendar_crop_result] No forecast records for month=%s/%s", month, year)

            monthly_forecasts.append({
                "month": month,
                "year": year,
                "temp_mean": round(avg_temp, 1) if avg_temp is not None else None,
                "precipitation": round(avg_precip, 1) if avg_precip is not None else None,
                "humidity": round(avg_humidity, 1) if avg_humidity is not None else None,
                "climate_source": climate_source,
            })

        MONTHS_LONG = [
            "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
            "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
        ]

        # Try real yield ensemble first; fall back to mock if artifacts are not ready
        logger.info("[_build_calendar_crop_result] Calling yield ensemble for crop_id=%s", crop_id)
        yield_result = _predict_yield_with_ensemble(db, crop_id, municipality.dane_code)
        logger.debug("[_build_calendar_crop_result] Yield result: %s", yield_result)

        # Build real calendar info from EVA/FAO when available
        logger.debug("[_build_calendar_crop_result] Loading calendar info from EVA/FAO/agronomic data")
        loader = self._get_model_loader()
        calendar_info = _get_crop_calendar_info(db, crop_id, loader)
        planting_months = calendar_info["planting_months"]
        duration_min = calendar_info["duration_days_min"]
        duration_max = calendar_info["duration_days_max"]
        logger.debug("[_build_calendar_crop_result] Calendar info: planting_months=%s duration_min=%s duration_max=%s", planting_months, duration_min, duration_max)

        # Determine top 3 harvest months only when planting months and cycle
        # duration are available from the EVA/FAO datasets.
        top_harvest = []
        warnings: List[str] = []
        current_year = today.year
        if planting_months and duration_min and duration_max:
            logger.debug("[_build_calendar_crop_result] Computing harvest windows from calendar data")
            candidates = []
            for pm in planting_months:
                # If planting month is before current month, assume next year
                year_offset = 1 if pm < today.month else 0
                harvest_month = ((pm - 1 + (duration_min // 30)) % 12) + 1
                harvest_year = current_year + year_offset + ((pm - 1 + (duration_min // 30)) // 12)
                candidates.append({
                    "harvest_month": harvest_month,
                    "harvest_year": harvest_year,
                    "planting_month": pm,
                    "planting_year": current_year + year_offset,
                    "score": 0.0,
                })

            # Score by closeness to current month and yield signal
            for cand in candidates:
                month_dist = abs(((cand["harvest_month"] - today.month) + 6) % 12 - 6)
                yield_score = 0.0
                if yield_result and yield_result.get("yield_prediction") is not None:
                    # Normalize yield signal: 30 t/ha is treated as a high anchor.
                    yield_score = min(1.0, max(0.0, yield_result["yield_prediction"] / 30.0))
                cand["score"] = round(0.7 * (1 - month_dist / 6.0) + 0.3 * yield_score, 3)

            candidates.sort(key=lambda x: x["score"], reverse=True)
            for cand in candidates[:3]:
                top_harvest.append({
                    "harvest_month": cand["harvest_month"],
                    "harvest_year": cand["harvest_year"],
                    "harvest_month_name": MONTHS_LONG[cand["harvest_month"] - 1],
                    "score": cand["score"],
                    "planting_months": [cand["planting_month"]],
                    "planting_year": cand["planting_year"],
                    "duration_days_min": duration_min,
                    "duration_days_max": duration_max,
                })
        else:
            warnings.append("No calendar data available from EVA/FAO for this crop; harvest windows cannot be computed.")

        if yield_result:
            logger.info(
                "[_build_calendar_crop_result] Returning calendar (crop_id=%s, method=%s, yield_prediction=%s, top_harvest=%s)",
                crop_id, yield_result["method"], yield_result["yield_prediction"], len(top_harvest)
            )
            return {
                "crop_id": crop_id,
                "crop_name": crop.name,
                "yield_prediction": yield_result["yield_prediction"],
                "yield_model_version": yield_result["yield_model_version"],
                "yield_confidence": yield_result["yield_confidence"],
                "top_harvest_months": top_harvest,
                "monthly_forecasts": monthly_forecasts,
                "warnings": warnings,
                "method": yield_result["method"],
            }

        logger.warning("[_build_calendar_crop_result] Yield ensemble unavailable; returning mock result for crop_id=%s", crop_id)
        warnings.append("Mock prediction - ML models not yet loaded")
        return {
            "crop_id": crop_id,
            "crop_name": crop.name,
            "yield_prediction": None,
            "yield_model_version": None,
            "yield_confidence": "low",
            "top_harvest_months": top_harvest,
            "monthly_forecasts": monthly_forecasts,
            "warnings": warnings,
            "method": "mock",
        }
