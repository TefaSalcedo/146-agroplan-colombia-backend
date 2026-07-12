import httpx
from datetime import datetime, date, timezone
from app.config import get_settings
from app.logger import get_logger

settings = get_settings()
logger = get_logger("app.services.open_meteo")


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
    
    def get_current_weather(self, lat: float, lng: float) -> dict:
        """Get current weather from Open-Meteo Forecast API"""
        url = f"{self.base_url}/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lng,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
            "timezone": "America/Bogota"
        }

        logger.debug("[get_current_weather] Calling Open-Meteo (lat=%s, lng=%s)", lat, lng)
        with httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
            response = client.get(url, params=params)
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
            "fetched_at": datetime.now(timezone.utc).isoformat()
        }
    
    def get_historical_weather(
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
            "daily": "temperature_2m_mean,temperature_2m_min,temperature_2m_max,precipitation_sum,relative_humidity_2m_mean,uv_index_max,wind_speed_10m_max",
            "timezone": "America/Bogota"
        }

        logger.debug("[get_historical_weather] Calling Open-Meteo Archive (lat=%s, lng=%s)", lat, lng)
        with httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        return data

    def get_daily_forecast(
        self,
        lat: float,
        lng: float,
        days: int = 90
    ) -> list[dict]:
        """Get daily forecast from Open-Meteo Forecast API.

        Returns a list of daily records with temperature, precipitation, humidity,
        UV index and wind speed. One Open-Meteo call covers all requested days.
        """
        url = f"{self.base_url}/v1/forecast"
        daily_vars = (
            "temperature_2m_min,temperature_2m_max,temperature_2m_mean,"
            "precipitation_sum,relative_humidity_2m_mean,uv_index_max,wind_speed_10m_max"
        )
        params = {
            "latitude": lat,
            "longitude": lng,
            "daily": daily_vars,
            "forecast_days": days,
            "timezone": "America/Bogota"
        }

        logger.debug("[get_daily_forecast] Calling Open-Meteo Forecast (lat=%s, lng=%s, days=%s)", lat, lng, days)
        with httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        if not dates:
            return []

        records = []
        for idx, date_str in enumerate(dates):
            try:
                forecast_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue

            records.append({
                "forecast_date": forecast_date,
                "temp_min": daily.get("temperature_2m_min", [])[idx] if idx < len(daily.get("temperature_2m_min", [])) else None,
                "temp_max": daily.get("temperature_2m_max", [])[idx] if idx < len(daily.get("temperature_2m_max", [])) else None,
                "temp_mean": daily.get("temperature_2m_mean", [])[idx] if idx < len(daily.get("temperature_2m_mean", [])) else None,
                "precipitation": daily.get("precipitation_sum", [])[idx] if idx < len(daily.get("precipitation_sum", [])) else None,
                "humidity": daily.get("relative_humidity_2m_mean", [])[idx] if idx < len(daily.get("relative_humidity_2m_mean", [])) else None,
                "uv_index": daily.get("uv_index_max", [])[idx] if idx < len(daily.get("uv_index_max", [])) else None,
                "wind_speed": daily.get("wind_speed_10m_max", [])[idx] if idx < len(daily.get("wind_speed_10m_max", [])) else None,
            })

        return records

    def get_monthly_seasonal_forecast(
        self,
        lat: float,
        lng: float,
        months: int = 4,
    ) -> list[dict]:
        """Get monthly seasonal forecast from Open-Meteo Seasonal API.

        Returns a list of monthly records with mean temperature, precipitation
        and anomalies for the requested horizon. SEAS5 provides forecasts up to
        7 months ahead and is updated monthly.
        """
        url = "https://seasonal-api.open-meteo.com/v1/seasonal"
        params = {
            "latitude": lat,
            "longitude": lng,
            "models": "ecmwf_seas5",
            "monthly": (
                "temperature_2m_mean,temperature_2m_anomaly,"
                "precipitation_mean,precipitation_anomaly"
            ),
            "forecast_days": months * 30,
            "timezone": "America/Bogota",
        }

        logger.debug(
            "[get_monthly_seasonal_forecast] Calling Open-Meteo Seasonal (lat=%s, lng=%s, months=%s)",
            lat, lng, months,
        )
        with httpx.Client(timeout=httpx.Timeout(20.0, connect=5.0)) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        monthly = data.get("monthly", {})
        dates = monthly.get("time", [])
        if not dates:
            return []

        records = []
        for idx, date_str in enumerate(dates):
            try:
                parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
                # Open-Meteo returns the last day of the month in the requested
                # timezone. Normalize to the first day of the month for clarity.
                forecast_month = parsed.replace(day=1)
            except (ValueError, TypeError):
                continue

            records.append({
                "forecast_month": forecast_month,
                "temp_mean": monthly.get("temperature_2m_mean", [])[idx] if idx < len(monthly.get("temperature_2m_mean", [])) else None,
                "temp_anomaly": monthly.get("temperature_2m_anomaly", [])[idx] if idx < len(monthly.get("temperature_2m_anomaly", [])) else None,
                "precipitation": monthly.get("precipitation_mean", [])[idx] if idx < len(monthly.get("precipitation_mean", [])) else None,
                "precipitation_anomaly": monthly.get("precipitation_anomaly", [])[idx] if idx < len(monthly.get("precipitation_anomaly", [])) else None,
                "source": "open-meteo-seasonal",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })

        return records
