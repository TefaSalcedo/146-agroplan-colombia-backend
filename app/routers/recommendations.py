from datetime import datetime, timezone

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.crop import CropResponseLite, TopCropResponse
from app.schemas.recommendation import (
    NextPlantingSeason,
    RecommendationRequest,
    RecommendationResponse,
)
from app.schemas.system import ErrorResponse
from app.services.crop_catalog import CropCatalog
from app.services.mock_predictor import MockPredictor
from app.services.municipality_catalog import MunicipalityCatalog

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
municipality_catalog = MunicipalityCatalog()
crop_catalog = CropCatalog()
predictor = MockPredictor()

MONTHS_LONG = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]


@router.post(
    "",
    response_model=RecommendationResponse,
    summary="Get crop recommendations",
    description=(
        "Ranks crops for a municipality using zoning suitability and confidence scores.\n\n"
        "Use cases:\n"
        "- Show a best-crop recommendation card.\n"
        "- Offer alternatives and the next planting season preview."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Municipality ID not found.",
        }
    },
)
def get_recommendations(
    request: RecommendationRequest = Body(
        ...,
        examples={
            "municipality_recommendation": {
                "summary": "Recommendations for one municipality",
                "value": {"municipality_id": "05001"},
            }
        },
    ),
    db: Session = Depends(get_db),
):
    municipality = municipality_catalog.get_municipality_by_id(db, request.municipality_id)
    if not municipality:
        raise HTTPException(status_code=404, detail="Municipality not found")

    all_crops = crop_catalog.get_all_crops(db)

    crop_scores = []
    for crop in all_crops:
        prediction = predictor.predict_zoning(
            db=db,
            crop_id=crop.id,
            municipality_id=request.municipality_id,
        )

        suitability_score = {
            "high": 3,
            "medium": 2,
            "low": 1,
            "none": 0,
        }.get(prediction["suitability"], 0)

        score = suitability_score * prediction["confidence"]
        crop_scores.append((crop, score, prediction))

    crop_scores.sort(key=lambda x: x[1], reverse=True)

    top_crop, _, top_prediction = crop_scores[0]

    top_crop_dict = top_crop.model_dump()
    top_crop_dict["suitability"] = top_prediction["suitability"]
    top_crop = TopCropResponse(**top_crop_dict)

    other_crops = [
        CropResponseLite(
            id=crop.id,
            name=crop.name,
            image=crop.image,
            recommendation=crop.recommendation,
            success_rate=crop.success_rate,
        )
        for crop, _, _ in crop_scores[1:5]
    ]

    current_month = datetime.now(timezone.utc).month
    next_month = current_month % 12 + 1

    plantable_crops = [
        crop.id for crop in all_crops
        if next_month in (crop.planting_months or [])
    ]

    next_planting_season = NextPlantingSeason(
        month=next_month,
        month_name=MONTHS_LONG[next_month - 1],
        crops=plantable_crops[:4],
    )

    return RecommendationResponse(
        top_crop=top_crop,
        other_crops=other_crops,
        next_planting_season=next_planting_season,
    )
