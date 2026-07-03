from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.municipality_catalog import MunicipalityCatalog
from app.services.open_meteo import OpenMeteoService
from app.schemas.weather import WeatherResponse

router = APIRouter(prefix="/weather", tags=["weather"])
municipality_catalog = MunicipalityCatalog()
weather_service = OpenMeteoService()


@router.get("/{municipality_id}", response_model=WeatherResponse)
async def get_weather(municipality_id: str, db: Session = Depends(get_db)):
    """Get current weather for a municipality"""
    municipality = municipality_catalog.get_municipality_by_id(db, municipality_id)
    if not municipality:
        raise HTTPException(status_code=404, detail="Municipality not found")
    
    weather_data = await weather_service.get_current_weather(
        lat=municipality.lat,
        lng=municipality.lng
    )
    
    return WeatherResponse(**weather_data)
