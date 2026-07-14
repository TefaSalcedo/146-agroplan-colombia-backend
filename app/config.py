from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://agroplan:agroplan@db:5432/agroplan"
    migration_database_url: str = ""

    # Open-Meteo
    open_meteo_base_url: str = "https://api.open-meteo.com"
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com"
    open_meteo_max_retries: int = 3
    open_meteo_retry_backoff_base: float = 1.0

    # ML Data & Models
    ml_data_path: str = "./data"
    ml_models_path: str = "./models"

    # Hugging Face
    hf_token: str = ""
    hf_model_repo_zoning: str = ""
    hf_model_repo_yield: str = ""
    hf_model_revision: str = "main"
    hf_download_mode: str = "mvp"  # mvp | all | none

    # LLM
    llm_enabled: bool = True
    llm_provider: str = "openrouter"
    openrouter_api_key: str = ""
    openrouter_models: str = ""
    groq_api_key: str = ""
    groq_models: str = ""
    cerebras_api_key: str = ""
    cerebras_models: str = ""
    nvidia_api_key: str = ""
    nvidia_models: str = ""
    llm_timeout_seconds: int = 30

    # Admin
    admin_api_key: str = ""

    # Cache
    cache_ttl_strategy: str = "end_of_month_bogota"

    # API
    api_v1_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:3000,http://localhost:3001"
    max_request_body_bytes: int = 65_536

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_ml_llm_per_minute: int = 20
    rate_limit_general_per_minute: int = 100
    rate_limit_window_seconds: int = 60

    # Climate Sync Job
    enable_climate_sync: bool = False
    climate_sync_hour: int = 3
    climate_sync_minute: int = 0
    climate_sync_batch_size: int = 100
    climate_sync_delay_seconds: float = 2.0
    climate_sync_days_ahead: int = 16
    climate_sync_cleanup_days: int = 180

    # Fallback predictor (development only)
    enable_mock_predictor: bool = False  # True allows MockPredictor fallback

    # Observability / Debug logging
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def openrouter_models_list(self) -> list[str]:
        return [m.strip() for m in self.openrouter_models.split(",") if m.strip()]

    @property
    def groq_models_list(self) -> list[str]:
        return [m.strip() for m in self.groq_models.split(",") if m.strip()]

    @property
    def cerebras_models_list(self) -> list[str]:
        return [m.strip() for m in self.cerebras_models.split(",") if m.strip()]

    @property
    def nvidia_models_list(self) -> list[str]:
        return [m.strip() for m in self.nvidia_models.split(",") if m.strip()]

    @property
    def effective_migration_url(self) -> str:
        return self.migration_database_url or self.database_url


@lru_cache()
def get_settings() -> Settings:
    return Settings()
