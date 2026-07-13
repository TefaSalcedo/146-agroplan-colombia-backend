import socket
import time
import httpx
from datetime import datetime, date, timezone
from typing import Optional

from app.config import get_settings
from app.logger import get_logger

settings = get_settings()
logger = get_logger("app.services.open_meteo")

# Exceptions that indicate a transient network/DNS issue worth retrying.
_RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    socket.gaierror,
    ConnectionError,
)


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
        self.max_retries = settings.open_meteo_max_retries
        self.backoff_base = settings.open_meteo_retry_backoff_base

    def _request_with_retry(
        self,
        url: str,
        params: dict,
        timeout: httpx.Timeout,
        label: str = "",
    ) -> httpx.Response:
        """Execute an HTTP GET with automatic retries on transient network errors.

        Retries on DNS failures, connection errors and timeouts using
        exponential backoff. HTTP error responses (4xx/5xx) are **not**
        retried because they usually indicate a permanent client-side issue
        (bad URL, bad parameters, rate limit, etc.).
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.get(url, params=params)
                    response.raise_for_status()
                    return response
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self.backoff_base * (2 ** (attempt - 1))
                    logger.warning(
                        "[%s] Attempt %s/%s failed (%s). Retrying in %.1fs",
                        label, attempt, self.max_retries, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "[%s] All %s attempts exhausted: %s",
                        label, self.max_retries, exc,
                    )
            except httpx.HTTPStatusError:
                raise
            except Exception as exc:
                last_exc = exc
                logger.error("[%s] Non-retryable error: %s", label, exc)
                raise
        raise last_exc  # type: ignore[misc]

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
        response = self._request_with_retry(
            url, params, httpx.Timeout(15.0, connect=5.0), label="get_current_weather",
        )
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
        response = self._request_with_retry(
            url, params, httpx.Timeout(15.0, connect=5.0), label="get_historical_weather",
        )
        data = response.json()
        return data

    def get_daily_forecast(
        self,
        lat: float,
        lng: float,
        days: int | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict]:
        """Get daily forecast from Open-Meteo Forecast API.

        Returns a list of daily records with temperature, precipitation, humidity,
        UV index and wind speed. Either ``days`` or both ``start_date`` and
        ``end_date`` must be provided. When ``start_date``/``end_date`` are given,
        the API is constrained to that exact interval, which avoids returning
        dates before today in non-UTC timezones.
        """
        url = f"{self.base_url}/v1/forecast"
        daily_vars = (
            "temperature_2m_min,temperature_2m_max,temperature_2m_mean,"
            "precipitation_sum,relative_humidity_2m_mean,uv_index_max,wind_speed_10m_max"
        )
        params: dict[str, str | int | float] = {
            "latitude": lat,
            "longitude": lng,
            "daily": daily_vars,
            "timezone": "America/Bogota",
        }

        if start_date and end_date:
            params["start_date"] = start_date.isoformat()
            params["end_date"] = end_date.isoformat()
            logger.debug(
                "[get_daily_forecast] Calling Open-Meteo Forecast (lat=%s, lng=%s, %s to %s)",
                lat, lng, start_date, end_date,
            )
        elif days is not None:
            params["forecast_days"] = days
            logger.debug("[get_daily_forecast] Calling Open-Meteo Forecast (lat=%s, lng=%s, days=%s)", lat, lng, days)
        else:
            raise ValueError("Either days or both start_date and end_date must be provided")

        response = self._request_with_retry(
            url, params, httpx.Timeout(15.0, connect=5.0), label="get_daily_forecast",
        )
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
        response = self._request_with_retry(
            url, params, httpx.Timeout(20.0, connect=5.0), label="get_monthly_seasonal_forecast",
        )
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

    def get_elevation(self, lat: float, lng: float) -> Optional[float]:
        """Get elevation from Open-Meteo elevation API."""
        url = f"{self.base_url}/v1/elevation"
        params = {"latitude": lat, "longitude": lng}
        try:
            logger.debug("[get_elevation] Calling Open-Meteo (lat=%s, lng=%s)", lat, lng)
            response = self._request_with_retry(
                url, params, httpx.Timeout(15.0, connect=5.0), label="get_elevation",
            )
            data = response.json()
            elevation = data.get("elevation")
            if isinstance(elevation, list) and elevation:
                elevation = elevation[0]
            if elevation is not None:
                return float(elevation)
        except Exception as e:
            logger.warning("[get_elevation] Failed: %s", e)
        return None

    def get_annual_climate(self, lat: float, lng: float) -> Optional[dict]:
        """Get annual average temperature and precipitation from Open-Meteo archive.

        Uses the last 10 years of historical data to compute long-term averages.
        """
        end_year = datetime.now(timezone.utc).year - 1
        start_year = end_year - 9
        url = f"{self.archive_url}/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lng,
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "America/Bogota",
        }
        try:
            logger.debug(
                "[get_annual_climate] Calling Open-Meteo Archive (lat=%s, lng=%s, %s-%s)",
                lat, lng, start_year, end_year,
            )
            response = self._request_with_retry(
                url, params, httpx.Timeout(30.0, connect=5.0), label="get_annual_climate",
            )
            data = response.json()
        except Exception as e:
            logger.warning("[get_annual_climate] Failed: %s", e)
            return None

        daily = data.get("daily", {})
        temps = daily.get("temperature_2m_mean", [])
        precips = daily.get("precipitation_sum", [])

        valid_temps = [t for t in temps if t is not None]
        valid_precips = [p for p in precips if p is not None]

        if not valid_temps or not valid_precips:
            return None

        avg_temp = round(sum(valid_temps) / len(valid_temps), 1)
        annual_precip = round(sum(valid_precips) / len(valid_precips) * 365, 1)

        return {
            "avg_temperature": avg_temp,
            "precipitation": annual_precip,
            "years": [start_year, end_year],
            "source": "open-meteo-archive",
        }
