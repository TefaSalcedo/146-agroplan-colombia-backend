from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.municipality_catalog import MunicipalityCatalog
from app.services.mock_predictor import MockPredictor
from app.schemas.zoning import ZoningRequest, ZoningResponse

router = APIRouter(prefix="/zoning", tags=["zoning"])
municipality_catalog = MunicipalityCatalog()
predictor = MockPredictor()


@router.post("/predict", response_model=ZoningResponse)
def predict_zoning(request: ZoningRequest, db: Session = Depends(get_db)):
    """Predict zoning suitability for a crop in a municipality (mock)"""
    municipality = municipality_catalog.get_municipality_by_id(db, request.municipality_id)
    if not municipality:
        raise HTTPException(status_code=404, detail="Municipality not found")
    
    prediction = predictor.predict_zoning(
        crop_id=request.crop_id,
        municipality_id=request.municipality_id,
        municipality_altitude=municipality.altitude,
        municipality_avg_temp=municipality.avg_temperature or 20
    )
    
    return ZoningResponse(**prediction)
