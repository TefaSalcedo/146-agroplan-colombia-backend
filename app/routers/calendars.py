from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.municipality_catalog import MunicipalityCatalog
from app.services.crop_catalog import CropCatalog
from app.services.mock_predictor import MockPredictor
from app.schemas.calendar import CalendarRequest, CalendarResponse

router = APIRouter(prefix="/calendars", tags=["calendars"])
municipality_catalog = MunicipalityCatalog()
crop_catalog = CropCatalog()
predictor = MockPredictor()


@router.post("/predict", response_model=CalendarResponse)
def predict_calendar(request: CalendarRequest, db: Session = Depends(get_db)):
    """Predict calendar planting ratings for a crop in a municipality (mock)"""
    municipality = municipality_catalog.get_municipality_by_id(db, request.municipality_id)
    if not municipality:
        raise HTTPException(status_code=404, detail="Municipality not found")
    
    crop = crop_catalog.get_crop_by_id(request.crop_id)
    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found")
    
    prediction = predictor.predict_calendar(
        crop_id=request.crop_id,
        municipality_id=request.municipality_id,
        month=request.month,
        year=request.year,
        planting_months=crop.planting_months
    )
    
    return CalendarResponse(**prediction)
