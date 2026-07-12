from datetime import datetime, timezone

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.logger import get_logger
from app.schemas.crop import CropResponseLite, GrowthStage, Tip, TopCropResponse
from app.schemas.zoning import ClimateBasedRecommendation
from app.schemas.recommendation import (
    NextPlantingSeason,
    RecommendationRequest,
    RecommendationResponse,
)
from app.schemas.system import ErrorResponse
from app.services.crop_catalog import CropCatalog
from app.services.municipality_catalog import MunicipalityCatalog
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
municipality_catalog = MunicipalityCatalog()
crop_catalog = CropCatalog()
prediction_service = PredictionService()
logger = get_logger("app.routers.recommendations")

MONTHS_LONG = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]


@router.post(
    "",
    response_model=RecommendationResponse,
    summary="Get crop recommendations for a municipality",
    description=(
        "Ranks crops for a municipality using the production LightGBM zoning model.\n\n"
        "The backend evaluates every ML-supported crop for the given municipality, scores each one "
        "by suitability and confidence, and returns the best crop plus alternatives. "
        "Only municipality_id is required; crop selection is handled server-side.\n\n"
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
    logger.info("[endpoint] POST /recommendations called (municipality_id=%s)", request.municipality_id)

    logger.debug("[endpoint] Querying database for municipality_id=%s", request.municipality_id)
    municipality = municipality_catalog.get_municipality_by_id(db, request.municipality_id)
    if not municipality:
        logger.warning("[endpoint] Municipality not found: %s", request.municipality_id)
        raise HTTPException(status_code=404, detail="Municipality not found")

    logger.debug("[endpoint] Fetching all ML-supported crops")
    crops = crop_catalog.get_ml_supported_crops(db)
    logger.info("[endpoint] Calling PredictionService.predict_zoning for %s crops", len(crops))

    crop_scores = []
    for crop in crops:
        prediction = prediction_service.predict_zoning(
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

    top_crop_dict = {
        "id": top_crop.id,
        "name": top_crop.name,
        "image": top_crop.image or "",
        "scientific_name": top_crop.scientific_name or "",
        "success_rate": top_crop.success_rate or 0,
        "recommendation": top_crop.recommendation or "medium",
        "short_reason": top_crop.short_reason or "",
        "reason": top_crop.reason or "",
        "days_to_harvest": top_crop.days_to_harvest or 0,
        "soil_type": top_crop.soil_type or "",
        "ideal_temperature": top_crop.ideal_temperature or "",
        "humidity": top_crop.humidity or "",
        "precipitation": top_crop.precipitation or "",
        "altitude": top_crop.altitude or "",
        "irrigation": top_crop.irrigation or "",
        "substrates": top_crop.substrates or [],
        "planting_months": top_crop.planting_months or [],
        "harvest_months": top_crop.harvest_months or [],
        "stages": [GrowthStage(**s) for s in (top_crop.stages or [])],
        "tips": [Tip(**t) for t in (top_crop.tips or [])],
        "suitability": top_prediction["suitability"],
    }
    top_crop_response = TopCropResponse(**top_crop_dict)

    other_crops = [
        CropResponseLite(
            id=crop.id,
            name=crop.name,
            image=crop.image or "",
            recommendation=crop.recommendation or "medium",
            success_rate=crop.success_rate or 0,
        )
        for crop, _, _ in crop_scores[1:5]
    ]

    current_month = datetime.now(timezone.utc).month
    next_month = current_month % 12 + 1

    plantable_crops = [
        crop.id for crop in crops
        if next_month in (crop.planting_months or [])
    ]

    next_planting_season = NextPlantingSeason(
        month=next_month,
        month_name=MONTHS_LONG[next_month - 1],
        crops=plantable_crops[:4],
    )

    climate_recs = prediction_service.get_climate_analog_recommendations(db, request.municipality_id)
    logger.info(
        "[endpoint] POST /recommendations adding %s climate-based recommendations",
        len(climate_recs),
    )

    logger.info("[endpoint] POST /recommendations returning top_crop=%s", top_crop_response.id)
    return RecommendationResponse(
        top_crop=top_crop_response,
        other_crops=other_crops,
        climate_based_recommendations=[ClimateBasedRecommendation(**r) for r in climate_recs],
        next_planting_season=next_planting_season,
    )
