from fastapi import APIRouter, Depends, HTTPException, Body, Path, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.crop_catalog import CropCatalog
from app.services.municipality_catalog import MunicipalityCatalog
from app.services.mock_predictor import MockPredictor
from app.services.prediction_service import PredictionService
from app.schemas.zoning import (
    ZoningRequest,
    ZoningResponse,
    ZoningBatchRequest,
    ZoningBatchResponse,
    ZoningBatchCropResult,
    ZoningMapResponse,
    ZoningMapMunicipalityResult,
)
from app.schemas.system import ErrorResponse

router = APIRouter(prefix="/zoning", tags=["zoning"])
municipality_catalog = MunicipalityCatalog()
crop_catalog = CropCatalog()
mock_predictor = MockPredictor()
prediction_service = PredictionService()


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
    municipality = municipality_catalog.get_municipality_by_id(db, request.municipality_id)
    if not municipality:
        raise HTTPException(status_code=404, detail="Municipality not found")

    crop = crop_catalog.get_crop_by_id(db, request.crop_id)
    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found")

    result = prediction_service.predict_zoning(
        db=db,
        crop_id=request.crop_id,
        municipality_id=request.municipality_id,
    )

    return ZoningResponse(**result)


@router.post(
    "/recommendations",
    response_model=ZoningBatchResponse,
    summary="Get zoning recommendations for all crops",
    description=(
        "Evaluates all ML-supported crops for a municipality and returns them ranked.\n\n"
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
    municipality = municipality_catalog.get_municipality_by_id(db, request.municipality_id)
    if not municipality:
        raise HTTPException(status_code=404, detail="Municipality not found")

    crop_ids = request.crop_ids
    if not crop_ids:
        ml_crops = crop_catalog.get_ml_supported_crops(db)
        crop_ids = [c.id for c in ml_crops]

    results = []
    for crop_id in crop_ids:
        crop = crop_catalog.get_crop_model_by_id(db, crop_id)
        if not crop:
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

    # Sort by confidence descending
    results.sort(key=lambda x: x.confidence, reverse=True)

    return ZoningBatchResponse(
        municipality_id=request.municipality_id,
        municipality_name=municipality.name,
        results=results,
        model_version="mock-v1",
    )


@router.get(
    "/map/{crop_id}",
    response_model=ZoningMapResponse,
    summary="Get zoning map for a crop",
    description=(
        "Evaluates all municipalities for a crop and returns suitability scores.\n\n"
        "Uses CatBoost when its manifest is satisfied and k-NN as controlled fallback. "
        "The full monthly payload is cached. Returns all municipalities; "
        "the frontend joins results with DANE geometry for rendering."
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
    crop = crop_catalog.get_crop_model_by_id(db, crop_id)
    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found")

    from app.models import Municipality, Department

    municipalities = db.query(Municipality).order_by(Municipality.dane_code).all()

    results = []
    primary_method = "primary_model"

    for muni in municipalities:
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

    return ZoningMapResponse(
        crop_id=crop_id,
        crop_name=crop.name,
        model_version="mock-v1",
        method=primary_method,
        results=results,
        total_municipalities=len(results),
    )
