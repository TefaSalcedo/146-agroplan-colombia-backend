from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.municipality_catalog import MunicipalityCatalog
from app.services.crop_catalog import CropCatalog
from app.services.mock_predictor import MockPredictor
from app.schemas.recommendation import RecommendationRequest, RecommendationResponse, NextPlantingSeason
from app.schemas.crop import CropResponse, CropResponseLite
from datetime import datetime
import random

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
municipality_catalog = MunicipalityCatalog()
crop_catalog = CropCatalog()
predictor = MockPredictor()

MONTHS_LONG = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]


@router.post("", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest, db: Session = Depends(get_db)):
    """Get crop recommendations for a municipality"""
    municipality = municipality_catalog.get_municipality_by_id(db, request.municipality_id)
    if not municipality:
        raise HTTPException(status_code=404, detail="Municipality not found")
    
    # Get all crops
    all_crops = crop_catalog.get_all_crops()
    
    # Predict zoning for each crop
    crop_scores = []
    for crop in all_crops:
        prediction = predictor.predict_zoning(
            crop_id=crop.id,
            municipality_id=request.municipality_id,
            municipality_altitude=municipality.altitude,
            municipality_avg_temp=municipality.avg_temperature or 20
        )
        
        # Score: high=3, medium=2, low=1, none=0, weighted by confidence
        suitability_score = {
            "high": 3,
            "medium": 2,
            "low": 1,
            "none": 0
        }.get(prediction["suitability"], 0)
        
        score = suitability_score * prediction["confidence"]
        crop_scores.append((crop, score, prediction))
    
    # Sort by score descending
    crop_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Top crop
    top_crop, top_score, top_prediction = crop_scores[0]
    
    # Add suitability to top crop
    top_crop_dict = top_crop.model_dump()
    top_crop_dict["suitability"] = top_prediction["suitability"]
    top_crop = CropResponse(**top_crop_dict)
    
    # Other crops (top 4 excluding the top one)
    other_crops = [
        CropResponseLite(
            id=crop.id,
            name=crop.name,
            image=crop.image,
            recommendation=crop.recommendation,
            success_rate=crop.success_rate
        )
        for crop, _, _ in crop_scores[1:5]
    ]
    
    # Next planting season (current month + 1)
    current_month = datetime.now().month
    next_month = current_month % 12 + 1
    
    # Find crops that can be planted in next month
    plantable_crops = [
        crop.id for crop in all_crops
        if next_month in crop.planting_months
    ]
    
    next_planting_season = NextPlantingSeason(
        month=next_month,
        month_name=MONTHS_LONG[next_month - 1],
        crops=plantable_crops[:4]  # Limit to 4 crops
    )
    
    return RecommendationResponse(
        top_crop=top_crop,
        other_crops=other_crops,
        next_planting_season=next_planting_season
    )
