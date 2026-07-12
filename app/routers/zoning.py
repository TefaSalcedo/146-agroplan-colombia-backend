from fastapi import APIRouter, Depends, HTTPException, Path, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.logger import get_logger
from app.models import Municipality
from app.schemas.zoning import (
    ClimateBasedRecommendation,
    MunicipalityAIGuideResponse,
    ZoningBatchCropResult,
    ZoningBatchResponse,
    ZoningMapMunicipalityResult,
    ZoningMapResponse,
)
from app.schemas.system import ErrorResponse
from app.services.crop_catalog import CropCatalog
from app.services.municipality_ai_guide_service import get_municipality_ai_guide_service
from app.services.municipality_catalog import MunicipalityCatalog
from app.services.prediction_service import PredictionService
from app.middleware.rate_limit import ML_LLM_GROUP, enforce_rate_limit, rate_limit

router = APIRouter(prefix="/zoning", tags=["zoning"])
municipality_catalog = MunicipalityCatalog()
crop_catalog = CropCatalog()
prediction_service = PredictionService()
ai_guide_service = get_municipality_ai_guide_service()
logger = get_logger("app.routers.zoning")


@router.get(
    "/recommendations/{municipality_id}",
    response_model=ZoningBatchResponse,
    summary="Get crop recommendations for a municipality",
    description=(
        "Returns crops recommended for a municipality.\n\n"
        "The backend evaluates every ML-supported crop with the LightGBM zoning model (or fallback) "
        "and returns only those with suitability ``high`` or ``medium``. No crop_id is required.\n\n"
        "The response also includes an additional ``climate_based_recommendations`` list "
        "with crops recommended by the climate+soil k-NN analog model, excluding crops already "
        "recommended by LightGBM."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Municipality ID not found.",
        },
    },
)
@enforce_rate_limit
@rate_limit(ML_LLM_GROUP)
def get_zoning_recommendations_by_municipality(
    http_request: Request,
    municipality_id: str = Path(..., description="Municipality DANE code (5 digits)"),
    db: Session = Depends(get_db),
):
    logger.info("[endpoint] GET /zoning/recommendations/{municipality_id} called (municipality_id=%s)", municipality_id)

    logger.debug("[endpoint] Querying database for municipality_id=%s", municipality_id)
    municipality = municipality_catalog.get_municipality_by_id(db, municipality_id)
    if not municipality:
        logger.warning("[endpoint] Municipality not found: %s", municipality_id)
        raise HTTPException(status_code=404, detail="Municipality not found")

    logger.debug("[endpoint] Fetching all ML-supported crops")
    ml_crops = crop_catalog.get_ml_supported_crops(db)
    crop_ids = [c.id for c in ml_crops]
    logger.debug("[endpoint] Evaluating %s crops", len(crop_ids))

    results = []
    for crop_id in crop_ids:
        logger.debug("[endpoint] Evaluating crop_id=%s for municipality_id=%s", crop_id, municipality_id)
        crop = crop_catalog.get_crop_model_by_id(db, crop_id)
        if not crop:
            logger.warning("[endpoint] Crop not found, skipping: %s", crop_id)
            continue

        prediction = prediction_service.predict_zoning(
            db=db,
            crop_id=crop_id,
            municipality_id=municipality_id,
        )

        results.append(
            ZoningBatchCropResult(
                crop_id=crop_id,
                crop_name=crop.name,
                suitability=prediction["suitability"],
                confidence=prediction["confidence"],
                model_version=prediction["model_version"],
                method=prediction.get("method", "primary_model"),
                factors=prediction["factors"],
                probabilities=prediction.get("probabilities"),
                warnings=prediction.get("warnings"),
            )
        )

    # Keep only crops that LightGBM classified as high or medium.
    results = [r for r in results if r.suitability in ("high", "medium")]
    results.sort(key=lambda x: x.confidence, reverse=True)
    logger.info("[endpoint] GET /zoning/recommendations/{municipality_id} returning %s recommended crops", len(results))

    lightgbm_crop_ids = [r.crop_id for r in results]
    climate_recs = prediction_service.get_climate_analog_recommendations(
        db,
        municipality_id,
        exclude_crop_ids=lightgbm_crop_ids,
    )
    logger.info(
        "[endpoint] GET /zoning/recommendations/{municipality_id} adding %s climate-based recommendations",
        len(climate_recs),
    )

    return ZoningBatchResponse(
        municipality_id=municipality_id,
        municipality_name=municipality.name,
        results=results,
        climate_based_recommendations=[ClimateBasedRecommendation(**r) for r in climate_recs],
        model_version="zoning-lightgbm-v1",
    )


@router.get(
    "/map/{crop_id}",
    response_model=ZoningMapResponse,
    summary="Get zoning map for a crop",
    description=(
        "Evaluates all municipalities for a crop and returns suitability scores.\n\n"
        "Uses the CatBoost zoning model with a single batch prediction for all "
        "municipalities. Only medium/high suitability results are returned to reduce noise. "
        "The frontend joins results with DANE geometry for rendering."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Crop ID not found.",
        },
    },
)
@enforce_rate_limit
@rate_limit(ML_LLM_GROUP)
def get_zoning_map(
    http_request: Request,
    crop_id: str = Path(..., description="Crop identifier (e.g. aguacate)"),
    db: Session = Depends(get_db),
):
    logger.info("[endpoint] GET /zoning/map/{crop_id} called (crop_id=%s)", crop_id)

    logger.debug("[endpoint] Querying database for crop_id=%s", crop_id)
    crop = crop_catalog.get_crop_model_by_id(db, crop_id)
    if not crop:
        logger.warning("[endpoint] Crop not found: %s", crop_id)
        raise HTTPException(status_code=404, detail="Crop not found")

    logger.info("[endpoint] Generating batch zoning map for crop_id=%s", crop_id)
    predictions_df = prediction_service.predict_zoning_map_batch(db, crop_id)

    if predictions_df is None or predictions_df.empty:
        logger.warning("[endpoint] No CatBatch predictions available for crop_id=%s", crop_id)
        raise HTTPException(status_code=503, detail="Zoning map model not available")

    # Build a lookup of municipality coordinates/names from the database
    municipality_ids = [str(r) for r in predictions_df["cod_dane_m"].tolist()]
    municipalities = {
        str(m.dane_code): m
        for m in db.query(Municipality).filter(Municipality.dane_code.in_(municipality_ids)).all()
    }

    results = []
    for _, row in predictions_df.iterrows():
        muni_id = str(row["cod_dane_m"])
        muni = municipalities.get(muni_id)
        if not muni:
            continue
        results.append(
            ZoningMapMunicipalityResult(
                municipality_id=muni_id,
                municipality_name=muni.name,
                dane_code=muni_id,
                lat=muni.lat,
                lng=muni.lng,
                suitability=row["suitability"],
                confidence=row["confidence"],
                method=row["method"],
                probabilities=row["probabilities"],
            )
        )

    logger.info(
        "[endpoint] GET /zoning/map/{crop_id} returning %s medium/high results (method=catboost_batch)",
        len(results),
    )
    return ZoningMapResponse(
        crop_id=crop_id,
        crop_name=crop.name,
        model_version="zoning-catboost-v1",
        method="catboost_batch",
        results=results,
        total_municipalities=len(results),
    )


@router.get(
    "/recommendations/{municipality_id}/ai-insights",
    response_model=MunicipalityAIGuideResponse,
    summary="Get AI insights for a municipality",
    description=(
        "Returns AI-generated insights for a municipality. Includes alternative crop "
        "recommendations, farming-system suggestions (viveros, hidroponia, etc.) and "
        "soil/fertilizer advice written in plain Spanish for farmers. The response is "
        "generated by an LLM and cached for 3 months."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Municipality ID not found.",
        },
    },
)
@enforce_rate_limit
@rate_limit(ML_LLM_GROUP)
def get_municipality_ai_guide(
    http_request: Request,
    municipality_id: str = Path(..., description="Municipality DANE code (5 digits)"),
    db: Session = Depends(get_db),
):
    logger.info(
        "[endpoint] GET /zoning/recommendations/{municipality_id}/ai-insights called (municipality_id=%s)",
        municipality_id,
    )
    municipality = municipality_catalog.get_municipality_by_id(db, municipality_id)
    if not municipality:
        logger.warning("[endpoint] Municipality not found: %s", municipality_id)
        raise HTTPException(status_code=404, detail="Municipality not found")

    guide = ai_guide_service.get_or_generate(db=db, municipality=municipality)
    logger.info(
        "[endpoint] GET /zoning/recommendations/{municipality_id}/ai-insights returning guide for municipality_id=%s",
        municipality_id,
    )
    return MunicipalityAIGuideResponse(**guide)
