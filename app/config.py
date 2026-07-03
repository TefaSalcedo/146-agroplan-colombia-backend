from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://agroplan:agroplan@db:5432/agroplan"

    # Open-Meteo
    open_meteo_base_url: str = "https://api.open-meteo.com"
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com"

    # ML Data Path
    ml_data_path: str = "./data"

    # API
    api_v1_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:3000,http://localhost:3001"

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
