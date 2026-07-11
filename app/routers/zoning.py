from fastapi import APIRouter, Depends, HTTPException, Body, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.crop_catalog import CropCatalog
from app.services.municipality_catalog import MunicipalityCatalog
from app.services.mock_predictor import MockPredictor
from app.schemas.zoning import (
    ZoningRequest,
    ZoningResponse,
    ZoningBatchRequest,
    ZoningBatchResponse,
)
from app.schemas.system import ErrorResponse

router = APIRouter(prefix="/zoning", tags=["zoning"])
municipality_catalog = MunicipalityCatalog()
crop_catalog = CropCatalog()
predictor = MockPredictor()


@router.post(
    "/predict",
    response_model=ZoningResponse,
    summary="Predict zoning suitability",
    description=(
        "Returns crop suitability for a municipality using the current mock predictor.\n\n"
        "Use cases:\n"
        "- Evaluate if a crop is viable in a selected municipality.\n"
        "- Explain factors (temperature, precipitation, soil, altitude) behind suitability."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Crop or municipality ID not found.",
        }
    },
)
def predict_zoning(
    request: ZoningRequest = Body(
        ...,
        examples={
            "coffee_manizales": {
                "summary": "Coffee in a mountain municipality",
                "value": {"crop_id": "cafe", "municipality_id": "17001"},
            }
        },
    ),
    db: Session = Depends(get_db),
):
    municipality = municipality_catalog.get_municipality_by_id(db, request.municipality_id)
    if not municipality:
        raise HTTPException(status_code=404, detail="Municipality not found")

    crop = crop_catalog.get_crop_by_id(request.crop_id)
    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found")

    prediction = predictor.predict_zoning(
        db=db,
        crop_id=request.crop_id,
        municipality_id=request.municipality_id,
    )

    return ZoningResponse(**prediction)


@router.post(
    "/predict/batch",
    response_model=ZoningBatchResponse,
    summary="Predict zoning suitability for all municipalities",
    description=(
        "Returns crop suitability predictions for every municipality covered by AgroPlan.\n\n"
        "Use cases:\n"
        "- Render a nationwide crop suitability map.\n"
        "- Compare viability across regions without issuing hundreds of single requests."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Crop ID not found.",
        }
    },
)
def predict_zoning_batch(
    request: ZoningBatchRequest,
    db: Session = Depends(get_db),
):
    crop = crop_catalog.get_crop_by_id(request.crop_id)
    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found")

    municipalities = municipality_catalog.get_municipalities(db)
    raw_predictions = predictor.predict_zoning_batch(
        db=db,
        crop_id=request.crop_id,
        municipalities=municipalities,
    )

    predictions = [ZoningResponse(**prediction) for prediction in raw_predictions]

    return ZoningBatchResponse(
        crop_id=request.crop_id,
        predictions=predictions,
        count=len(predictions),
        model_version="mock-v1",
    )
