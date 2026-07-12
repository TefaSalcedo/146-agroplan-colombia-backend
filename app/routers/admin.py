from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func, text

from app.database import get_db
from app.dependencies import verify_admin_api_key
from app.models import (
    MunicipalityClimateForecast,
    PredictionCache,
    ModelRelease,
    PredictionRun,
    LLMGeneration,
)
from app.schemas.admin import (
    ClimateSyncStatusResponse,
    ModelStatusResponse,
    ModelReleaseResponse,
    CacheStatsResponse,
    CacheInvalidateRequest,
    CacheInvalidateResponse,
    PredictionRunResponse,
)
from app.schemas.system import ErrorResponse
from app.services.climate_scheduler import get_last_sync_status
from app.services.model_loader import get_model_loader

router = APIRouter(prefix="/admin", tags=["admin"])

_UNAUTHORIZED_RESPONSE = {
    status.HTTP_401_UNAUTHORIZED: {
        "model": ErrorResponse,
        "description": "Missing or invalid admin API key.",
    }
}

_FORBIDDEN_RESPONSE = {
    status.HTTP_403_FORBIDDEN: {
        "model": ErrorResponse,
        "description": "Admin API key is not valid.",
    }
}

_ADMIN_ERROR_RESPONSES = {**_UNAUTHORIZED_RESPONSE, **_FORBIDDEN_RESPONSE}


@router.get(
    "/climate-sync/status",
    response_model=ClimateSyncStatusResponse,
    summary="Get climate sync status",
    description=(
        "Returns the latest climate synchronization run and forecast coverage counters.\n\n"
        "Use cases:\n"
        "- Monitor scheduled sync health from operations dashboards.\n"
        "- Verify data freshness before enabling weather-dependent features."
    ),
)
def climate_sync_status(db: Session = Depends(get_db)):
    last_sync = get_last_sync_status(db)

    forecast_count = db.query(MunicipalityClimateForecast).count()
    distinct_municipalities = (
        db.query(MunicipalityClimateForecast.municipality_dane_code)
        .distinct()
        .count()
    )

    return {
        "last_sync": last_sync,
        "forecast_records": forecast_count,
        "municipalities_with_forecast": distinct_municipalities,
    }


@router.get(
    "/models/status",
    response_model=ModelStatusResponse,
    summary="Get ML model status",
    description=(
        "Returns the status of all loaded ML models, profiles and golden vector validation.\n\n"
        "Use cases:\n"
        "- Verify that production models are ready before serving predictions.\n"
        "- Inspect active model releases and validation results."
    ),
    responses=_ADMIN_ERROR_RESPONSES,
    dependencies=[Depends(verify_admin_api_key)],
)
def model_status(db: Session = Depends(get_db)):
    loader = get_model_loader()

    # Get active releases from DB
    releases = db.query(ModelRelease).filter(ModelRelease.is_active == True).all()
    zoning_releases = [
        ModelReleaseResponse(
            id=r.id,
            model_type=r.model_type,
            crop_key=r.crop_key,
            model_family=r.model_family,
            hf_repo=r.hf_repo,
            hf_revision=r.hf_revision,
            artifact_filename=r.artifact_filename,
            is_active=r.is_active,
            sha256=r.sha256,
            preprocessor_version=r.preprocessor_version,
        )
        for r in releases if r.model_type == "zoning"
    ]
    yield_releases = [
        ModelReleaseResponse(
            id=r.id,
            model_type=r.model_type,
            crop_key=r.crop_key,
            model_family=r.model_family,
            hf_repo=r.hf_repo,
            hf_revision=r.hf_revision,
            artifact_filename=r.artifact_filename,
            is_active=r.is_active,
            sha256=r.sha256,
            preprocessor_version=r.preprocessor_version,
        )
        for r in releases if r.model_type == "yield"
    ]

    return ModelStatusResponse(
        models_loaded=loader.is_zoning_model_loaded(),
        zoning_models=zoning_releases,
        yield_models=yield_releases,
        profiles_loaded=loader.is_profiles_loaded(),
        golden_vectors_passed=loader.is_golden_vectors_passed(),
    )


@router.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Get prediction cache statistics",
    description=(
        "Returns counts of cache entries by type and expiry status.\n\n"
        "Use cases:\n"
        "- Monitor cache hit rates and memory pressure.\n"
        "- Decide when to invalidate stale entries."
    ),
    responses=_ADMIN_ERROR_RESPONSES,
    dependencies=[Depends(verify_admin_api_key)],
)
def cache_stats(db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)

    total = db.query(PredictionCache).count()
    active = db.query(PredictionCache).filter(PredictionCache.expires_at > now).count()
    expired = total - active

    by_type_rows = (
        db.query(PredictionCache.prediction_type, func.count(PredictionCache.id))
        .filter(PredictionCache.expires_at > now)
        .group_by(PredictionCache.prediction_type)
        .all()
    )
    by_type = {row[0]: row[1] for row in by_type_rows}

    return CacheStatsResponse(
        total_entries=total,
        active_entries=active,
        expired_entries=expired,
        by_type=by_type,
    )


@router.post(
    "/cache/invalidate",
    response_model=CacheInvalidateResponse,
    summary="Invalidate prediction cache",
    description=(
        "Deletes cache entries matching the optional filters.\n\n"
        "If no filters are provided, the entire prediction cache is cleared.\n\n"
        "Use cases:\n"
        "- Force model reload after a deployment.\n"
        "- Clear stale entries for a specific prediction type or scope."
    ),
    responses=_ADMIN_ERROR_RESPONSES,
    dependencies=[Depends(verify_admin_api_key)],
)
def invalidate_cache(
    request: CacheInvalidateRequest = Body(
        ...,
        examples={
            "invalidate_all": {
                "summary": "Clear all cache entries",
                "value": {},
            },
            "invalidate_by_type": {
                "summary": "Clear only zoning cache",
                "value": {"prediction_type": "zoning"},
            },
        },
    ),
    db: Session = Depends(get_db),
):
    query = db.query(PredictionCache)

    if request.prediction_type:
        query = query.filter(PredictionCache.prediction_type == request.prediction_type)
    if request.scope_key:
        query = query.filter(PredictionCache.scope_key == request.scope_key)

    count = query.delete()
    db.commit()

    return CacheInvalidateResponse(invalidated=count)


@router.get(
    "/audit/recent",
    response_model=List[PredictionRunResponse],
    summary="Get recent prediction runs",
    description=(
        "Returns the most recent prediction run entries for audit.\n\n"
        "Use cases:\n"
        "- Debug production failures and latency spikes.\n"
        "- Track fallback usage and cache effectiveness."
    ),
    responses=_ADMIN_ERROR_RESPONSES,
    dependencies=[Depends(verify_admin_api_key)],
)
def recent_prediction_runs(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of audit entries to return"),
    db: Session = Depends(get_db),
):
    runs = (
        db.query(PredictionRun)
        .order_by(PredictionRun.created_at.desc())
        .limit(limit)
        .all()
    )

    return [
        PredictionRunResponse(
            id=run.id,
            request_id=run.request_id,
            prediction_type=run.prediction_type,
            cache_hit=run.cache_hit,
            method=run.method,
            fallback_used=run.fallback_used,
            latency_ms=run.latency_ms,
            status=run.status,
            error_message=run.error_message,
            created_at=run.created_at.isoformat() if run.created_at else None,
        )
        for run in runs
    ]
