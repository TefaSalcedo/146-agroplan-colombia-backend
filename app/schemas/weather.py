from pydantic import BaseModel
from datetime import datetime


class WeatherResponse(BaseModel):
    temperature: float
    condition: str
    humidity: float
    precipitation: float
    icon: str
    source: str = "open-meteo"
    fetched_at: str
