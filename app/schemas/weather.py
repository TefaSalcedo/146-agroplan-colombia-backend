from pydantic import BaseModel, Field


class WeatherResponse(BaseModel):
    temperature: float = Field(description="Current air temperature in Celsius")
    condition: str = Field(description="Human-readable weather condition")
    humidity: float = Field(description="Relative humidity percentage")
    precipitation: float = Field(description="Current precipitation in millimeters")
    icon: str = Field(description="Frontend icon token")
    source: str = Field(default="open-meteo", description="Upstream provider")
    fetched_at: str = Field(description="UTC timestamp when weather was fetched")
