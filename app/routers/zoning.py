from fastapi import APIRouter, Depends, HTTPException, Body, Path, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.logger import get_logger
from app.services.crop_catalog import CropCatalog
from app.services.municipality_catalog import MunicipalityCatalog
from app.services.mock_predictor import MockPredictor
from app.services.prediction_service import PredictionService
from app.schemas.zoning import (
    ClimateBasedRecommendation,
    ZoningRequest,
    ZoningResponse,
    ZoningBatchRequest,
    ZoningBatchResponse,
    ZoningBatchCropResult,
    ZoningMapResponse,
    ZoningMapMunicipalityResult,
    ZoningMockBatchRequest,
    ZoningMockBatchResponse,
)
from app.schemas.system import ErrorResponse

router = APIRouter(prefix="/zoning", tags=["zoning"])
municipality_catalog = MunicipalityCatalog()
crop_catalog = CropCatalog()
mock_predictor = MockPredictor()
prediction_service = PredictionService()
logger = get_logger("app.routers.zoning")


@router.post(
    "/predict",
    response_model=ZoningResponse,
    summary="Predict zoning suitability",
    description=(
        "Returns crop suitability for a municipality.\n\n"
        "When ML models are loaded, uses LightGBM with 60 features. "
        "If critical features are missing, falls back to k-NN agroclimatic analogs. "
        "If no valid profile exists, returns 422."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Crop or municipality ID not found.",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "Critical features missing and no fallback available.",
        },
    },
)
def predict_zoning(
    request: ZoningRequest = Body(
        ...,
        examples={
            "aguacate_medellin": {
                "summary": "Avocado in Medellin",
                "value": {"crop_id": "aguacate", "municipality_id": "05001"},
            }
        },
    ),
    db: Session = Depends(get_db),
):
    logger.info("[endpoint] POST /zoning/predict called (crop_id=%s, municipality_id=%s)", request.crop_id, request.municipality_id)

    logger.debug("[endpoint] Querying database for municipality_id=%s", request.municipality_id)
    municipality = municipality_catalog.get_municipality_by_id(db, request.municipality_id)
    if not municipality:
        logger.warning("[endpoint] Municipality not found: %s", request.municipality_id)
        raise HTTPException(status_code=404, detail="Municipality not found")

    logger.debug("[endpoint] Querying database for crop_id=%s", request.crop_id)
    crop = crop_catalog.get_crop_by_id(db, request.crop_id)
    if not crop:
        logger.warning("[endpoint] Crop not found: %s", request.crop_id)
        raise HTTPException(status_code=404, detail="Crop not found")

    logger.info("[endpoint] Calling PredictionService.predict_zoning")
    result = prediction_service.predict_zoning(
        db=db,
        crop_id=request.crop_id,
        municipality_id=request.municipality_id,
    )
    logger.info("[endpoint] POST /zoning/predict returning (suitability=%s, confidence=%s, method=%s)", result.get("suitability"), result.get("confidence"), result.get("method"))

    return ZoningResponse(**result)


@router.get(
    "/recommendations/{municipality_id}",
    response_model=ZoningBatchResponse,
    summary="Get zoning recommendations for a municipality",
    description=(
        "Evaluates all ML-supported crops for a municipality and returns them ranked.\n\n"
        "The backend resolves the municipality and runs the LightGBM zoning model (or fallback) "
        "for every supported crop. No crop_id is required.\n\n"
        "Each crop indicates whether it used the primary model, a fallback, or was unavailable. "
        "No crops are filtered by threshold; the frontend decides what to display."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Municipality ID not found.",
        },
    },
)
def get_zoning_recommendations_by_municipality(
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

    results.sort(key=lambda x: x.confidence, reverse=True)
    logger.info("[endpoint] GET /zoning/recommendations/{municipality_id} returning %s ranked crops", len(results))

    climate_recs = prediction_service.get_climate_analog_recommendations(db, municipality_id)
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


@router.post(
    "/recommendations",
    response_model=ZoningBatchResponse,
    summary="Get zoning recommendations for selected crops",
    description=(
        "Evaluates the provided crop list for a municipality and returns them ranked.\n\n"
        "Use the GET /zoning/recommendations/{municipality_id} endpoint when you want the backend "
        "to evaluate every supported crop automatically. This POST endpoint is kept for cases where "
        "the frontend wants to evaluate a custom subset of crops."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Municipality ID not found.",
        },
    },
)
def get_zoning_recommendations(
    request: ZoningBatchRequest = Body(
        ...,
        examples={
            "medellin_all_crops": {
                "summary": "All crops for Medellin",
                "value": {"municipality_id": "05001"},
            }
        },
    ),
    db: Session = Depends(get_db),
):
    logger.info("[endpoint] POST /zoning/recommendations called (municipality_id=%s, crop_ids=%s)", request.municipality_id, request.crop_ids)

    logger.debug("[endpoint] Querying database for municipality_id=%s", request.municipality_id)
    municipality = municipality_catalog.get_municipality_by_id(db, request.municipality_id)
    if not municipality:
        logger.warning("[endpoint] Municipality not found: %s", request.municipality_id)
        raise HTTPException(status_code=404, detail="Municipality not found")

    crop_ids = request.crop_ids
    if not crop_ids:
        logger.debug("[endpoint] No crop_ids provided; fetching all ML-supported crops")
        ml_crops = crop_catalog.get_ml_supported_crops(db)
        crop_ids = [c.id for c in ml_crops]
        logger.debug("[endpoint] Evaluating %s crops", len(crop_ids))

    results = []
    for crop_id in crop_ids:
        logger.debug("[endpoint] Evaluating crop_id=%s for municipality_id=%s", crop_id, request.municipality_id)
        crop = crop_catalog.get_crop_model_by_id(db, crop_id)
        if not crop:
            logger.warning("[endpoint] Crop not found, skipping: %s", crop_id)
            continue

        prediction = prediction_service.predict_zoning(
            db=db,
            crop_id=crop_id,
            municipality_id=request.municipality_id,
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

    results.sort(key=lambda x: x.confidence, reverse=True)
    logger.info("[endpoint] POST /zoning/recommendations returning %s ranked crops", len(results))

    climate_recs = prediction_service.get_climate_analog_recommendations(db, request.municipality_id)
    logger.info(
        "[endpoint] POST /zoning/recommendations adding %s climate-based recommendations",
        len(climate_recs),
    )

    return ZoningBatchResponse(
        municipality_id=request.municipality_id,
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
        "Uses the primary LightGBM zoning model when artifacts are loaded; falls back to "
        "k-NN agroclimatic analogs when critical features are missing. The full payload is "
        "cached. Returns all municipalities; the frontend joins results with DANE geometry "
        "for rendering."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Crop ID not found.",
        },
    },
)
def get_zoning_map(
    crop_id: str = Path(..., description="Crop identifier (e.g. aguacate)"),
    db: Session = Depends(get_db),
):
    logger.info("[endpoint] GET /zoning/map/{crop_id} called (crop_id=%s)", crop_id)

    logger.debug("[endpoint] Querying database for crop_id=%s", crop_id)
    crop = crop_catalog.get_crop_model_by_id(db, crop_id)
    if not crop:
        logger.warning("[endpoint] Crop not found: %s", crop_id)
        raise HTTPException(status_code=404, detail="Crop not found")

    from app.models import Municipality

    logger.debug("[endpoint] Querying database for all municipalities")
    municipalities = db.query(Municipality).order_by(Municipality.dane_code).all()
    logger.info("[endpoint] Generating zoning map for %s municipalities", len(municipalities))

    results = []
    primary_method = "primary_model"

    for muni in municipalities:
        logger.debug("[endpoint] Predicting zoning for municipality_id=%s", muni.dane_code)
        prediction = prediction_service.predict_zoning(
            db=db,
            crop_id=crop_id,
            municipality_id=muni.dane_code,
        )
        method = prediction.get("method", "primary_model")
        if method != "primary_model" and primary_method == "primary_model":
            primary_method = method

        results.append(
            ZoningMapMunicipalityResult(
                municipality_id=muni.dane_code,
                municipality_name=muni.name,
                dane_code=muni.dane_code,
                lat=muni.lat,
                lng=muni.lng,
                suitability=prediction["suitability"],
                confidence=prediction["confidence"],
                method=method,
                probabilities=prediction.get("probabilities"),
            )
        )

    logger.info("[endpoint] GET /zoning/map/{crop_id} returning %s results (primary_method=%s)", len(results), primary_method)
    return ZoningMapResponse(
        crop_id=crop_id,
        crop_name=crop.name,
        model_version="mock-v1",
        method=primary_method,
        results=results,
        total_municipalities=len(results),
    )


@router.post(
    "/mock/predict/batch",
    response_model=ZoningMockBatchResponse,
    summary="[MOCK] Predict zoning suitability for all municipalities (not for MVP)",
    description=(
        "⚠️ MOCK ENDPOINT — NOT INTENDED FOR MVP PRODUCTION USE.\n\n"
        "Legacy mock endpoint that returns crop suitability predictions for every municipality "
        "using static heuristics instead of the production LightGBM model.\n\n"
        "Use cases:\n"
        "- Render a nationwide crop suitability map using the mock predictor.\n"
        "- Compare viability across regions without issuing hundreds of single requests.\n\n"
        "For real predictions, use POST /zoning/predict or POST /zoning/recommendations."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Crop ID not found.",
        }
    },
)
def predict_zoning_mock_batch(
    request: ZoningMockBatchRequest,
    db: Session = Depends(get_db),
):
    logger.info("[endpoint] POST /zoning/mock/predict/batch called (crop_id=%s)", request.crop_id)

    logger.debug("[endpoint] Querying database for crop_id=%s", request.crop_id)
    crop = crop_catalog.get_crop_by_id(db, request.crop_id)
    if not crop:
        logger.warning("[endpoint] Crop not found: %s", request.crop_id)
        raise HTTPException(status_code=404, detail="Crop not found")

    logger.debug("[endpoint] Querying database for all municipalities")
    municipalities = municipality_catalog.get_municipalities(db)
    logger.info("[endpoint] Calling MockPredictor.predict_zoning_batch for %s municipalities", len(municipalities))
    raw_predictions = mock_predictor.predict_zoning_batch(
        db=db,
        crop_id=request.crop_id,
        municipalities=municipalities,
    )

    predictions = [ZoningResponse(**prediction) for prediction in raw_predictions]
    logger.info("[endpoint] POST /zoning/mock/predict/batch returning %s predictions", len(predictions))

    return ZoningMockBatchResponse(
        crop_id=request.crop_id,
        predictions=predictions,
        count=len(predictions),
        model_version="mock-v1",
    )
