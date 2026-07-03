import httpx
from datetime import datetime
from app.config import get_settings

settings = get_settings()


# WMO Weather Code mapping to frontend icons and conditions
WEATHER_CODE_MAP = {
    # Clear
    0: {"condition": "Despejado", "icon": "sun"},
    1: {"condition": "Mayormente despejado", "icon": "sun"},
    
    # Partly cloudy
    2: {"condition": "Parcialmente nublado", "icon": "partly"},
    3: {"condition": "Nublado", "icon": "cloud"},
    
    # Fog
    45: {"condition": "Niebla", "icon": "cloud"},
    48: {"condition": "Niebla con escarcha", "icon": "cloud"},
    
    # Drizzle
    51: {"condition": "Llovizna ligera", "icon": "rain"},
    53: {"condition": "Llovizna moderada", "icon": "rain"},
    55: {"condition": "Llovizna intensa", "icon": "rain"},
    
    # Freezing drizzle
    56: {"condition": "Llovizna helada ligera", "icon": "rain"},
    57: {"condition": "Llovizna helada intensa", "icon": "rain"},
    
    # Rain
    61: {"condition": "Lluvia ligera", "icon": "rain"},
    63: {"condition": "Lluvia moderada", "icon": "rain"},
    65: {"condition": "Lluvia intensa", "icon": "rain"},
    
    # Freezing rain
    66: {"condition": "Lluvia helada ligera", "icon": "rain"},
    67: {"condition": "Lluvia helada intensa", "icon": "rain"},
    
    # Snow
    71: {"condition": "Nieve ligera", "icon": "cloud"},
    73: {"condition": "Nieve moderada", "icon": "cloud"},
    75: {"condition": "Nieve intensa", "icon": "cloud"},
    
    # Snow grains
    77: {"condition": "Granizo de nieve", "icon": "cloud"},
    
    # Rain showers
    80: {"condition": "Aguaceros ligeros", "icon": "rain"},
    81: {"condition": "Aguaceros moderados", "icon": "rain"},
    82: {"condition": "Aguaceros intensos", "icon": "rain"},
    
    # Snow showers
    85: {"condition": "Aguaceros de nieve ligeros", "icon": "cloud"},
    86: {"condition": "Aguaceros de nieve intensos", "icon": "cloud"},
    
    # Thunderstorm
    95: {"condition": "Tormenta", "icon": "rain"},
    96: {"condition": "Tormenta con granizo ligero", "icon": "rain"},
    99: {"condition": "Tormenta con granizo intenso", "icon": "rain"},
}


class OpenMeteoService:
    def __init__(self):
        self.base_url = settings.open_meteo_base_url
        self.archive_url = settings.open_meteo_archive_url
    
    def _map_weather_code(self, code: int) -> dict:
        """Map WMO weather code to condition and icon"""
        return WEATHER_CODE_MAP.get(code, {"condition": "Desconocido", "icon": "cloud"})
    
    async def get_current_weather(self, lat: float, lng: float) -> dict:
        """Get current weather from Open-Meteo Forecast API"""
        url = f"{self.base_url}/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lng,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
            "timezone": "America/Bogota"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        
        current = data.get("current", {})
        weather_code = current.get("weather_code", 0)
        weather_info = self._map_weather_code(weather_code)
        
        return {
            "temperature": round(current.get("temperature_2m", 0), 1),
            "condition": weather_info["condition"],
            "humidity": round(current.get("relative_humidity_2m", 0)),
            "precipitation": round(current.get("precipitation", 0), 1),
            "icon": weather_info["icon"],
            "source": "open-meteo",
            "fetched_at": datetime.utcnow().isoformat() + "Z"
        }
    
    async def get_historical_weather(
        self, 
        lat: float, 
        lng: float, 
        start_date: str, 
        end_date: str
    ) -> dict:
        """Get historical weather from Open-Meteo Archive API"""
        url = f"{self.archive_url}/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lng,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean,shortwave_radiation_sum",
            "timezone": "America/Bogota"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        
        return data
