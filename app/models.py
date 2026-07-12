from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Date,
    DateTime,
    Text,
    Boolean,
    ForeignKey,
    UniqueConstraint,
    Index,
    JSON,
)
from sqlalchemy.sql import func
from app.database import Base


# ---------------------------------------------------------------------------
# Territorial catalog
# ---------------------------------------------------------------------------

class Department(Base):
    __tablename__ = "departments"

    dane_code = Column(String(2), primary_key=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Municipality(Base):
    __tablename__ = "municipalities"

    dane_code = Column(String(5), primary_key=True)
    department_dane_code = Column(
        String(2), ForeignKey("departments.dane_code"), nullable=False, index=True
    )
    name = Column(String(150), nullable=False)
    municipality_type = Column(String(50), nullable=True)
    lat = Column(Float, nullable=False)
    lng = Column(Float, nullable=False)
    altitude = Column(Integer, nullable=True)
    avg_temperature = Column(Float, nullable=True)
    precipitation = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Backwards-compatible property aliases used by existing code
    @property
    def id(self) -> str:
        return self.dane_code

    @property
    def department(self) -> str:
        """Department name for backwards compatibility with old schemas."""
        # This is set by the catalog service when hydrating from DB
        return getattr(self, "_department_name", "")


class MunicipalityClimateForecast(Base):
    __tablename__ = "municipality_climate_forecasts"
    __table_args__ = (
        UniqueConstraint("municipality_dane_code", "forecast_date", name="uq_climate_forecast"),
        Index("ix_climate_forecast_date", "forecast_date"),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    municipality_dane_code = Column(
        String(5), ForeignKey("municipalities.dane_code"), nullable=False, index=True
    )
    forecast_date = Column(Date, nullable=False, index=True)
    temp_min = Column(Float, nullable=True)
    temp_max = Column(Float, nullable=True)
    temp_mean = Column(Float, nullable=True)
    precipitation = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    uv_index = Column(Float, nullable=True)
    wind_speed = Column(Float, nullable=True)
    fetched_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class MunicipalityMonthlyClimateForecast(Base):
    __tablename__ = "municipality_monthly_climate_forecasts"
    __table_args__ = (
        UniqueConstraint(
            "municipality_dane_code", "forecast_month",
            name="uq_monthly_climate_forecast",
        ),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    municipality_dane_code = Column(
        String(5), ForeignKey("municipalities.dane_code"), nullable=False, index=True
    )
    forecast_month = Column(Date, nullable=False, index=True)
    temp_mean = Column(Float, nullable=True)
    temp_anomaly = Column(Float, nullable=True)
    precipitation = Column(Float, nullable=True)
    precipitation_anomaly = Column(Float, nullable=True)
    trend = Column(String(50), nullable=True)
    source = Column(String(50), nullable=False, default="open-meteo-seasonal")
    fetched_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class MunicipalityCurrentWeather(Base):
    __tablename__ = "municipality_current_weather"
    __table_args__ = (
        UniqueConstraint("municipality_dane_code", name="uq_current_weather_municipality"),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    municipality_dane_code = Column(
        String(5), ForeignKey("municipalities.dane_code"), nullable=False, index=True
    )
    temperature = Column(Float, nullable=False)
    condition = Column(String(100), nullable=False)
    humidity = Column(Float, nullable=False)
    precipitation = Column(Float, nullable=False)
    icon = Column(String(50), nullable=False)
    source = Column(String(50), nullable=False, default="open-meteo")
    fetched_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class ClimateSyncLog(Base):
    __tablename__ = "climate_sync_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    sync_type = Column(String(50), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=False)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    records_processed = Column(Integer, nullable=False, default=0)
    records_failed = Column(Integer, nullable=False, default=0)
    status = Column(String(20), nullable=False)
    error_message = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Crop catalog
# ---------------------------------------------------------------------------

class Crop(Base):
    __tablename__ = "crops"

    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    scientific_name = Column(String(200), nullable=True)
    image = Column(String(255), nullable=True)
    days_to_harvest = Column(Integer, nullable=True)
    soil_type = Column(String(255), nullable=True)
    ideal_temperature = Column(String(100), nullable=True)
    humidity = Column(String(100), nullable=True)
    precipitation = Column(String(100), nullable=True)
    altitude = Column(String(100), nullable=True)
    irrigation = Column(String(255), nullable=True)
    substrates = Column(JSON, nullable=True)
    planting_months = Column(JSON, nullable=True)
    harvest_months = Column(JSON, nullable=True)
    stages = Column(JSON, nullable=True)
    tips = Column(JSON, nullable=True)
    # ML linkage
    ml_crop_key = Column(String(50), nullable=True)
    is_ml_supported = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# ML model releases
# ---------------------------------------------------------------------------

class ModelRelease(Base):
    __tablename__ = "model_releases"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_type = Column(String(50), nullable=False, index=True)
    crop_key = Column(String(50), nullable=True, index=True)
    hf_repo = Column(String(255), nullable=False)
    hf_revision = Column(String(100), nullable=False)
    artifact_filename = Column(String(255), nullable=False)
    sha256 = Column(String(64), nullable=True)
    model_family = Column(String(50), nullable=False)
    preprocessor_version = Column(String(50), nullable=True)
    manifest = Column(JSON, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


# ---------------------------------------------------------------------------
# Prediction cache
# ---------------------------------------------------------------------------

class PredictionCache(Base):
    __tablename__ = "prediction_cache"
    __table_args__ = (
        UniqueConstraint("cache_key", name="uq_prediction_cache_key"),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    cache_key = Column(String(500), nullable=False, index=True)
    prediction_type = Column(String(50), nullable=False, index=True)
    scope_key = Column(String(255), nullable=True)
    reference_month = Column(Date, nullable=True)
    algorithm_version = Column(String(100), nullable=True)
    payload = Column(JSON, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


# ---------------------------------------------------------------------------
# Prediction audit
# ---------------------------------------------------------------------------

class PredictionRun(Base):
    __tablename__ = "prediction_runs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    request_id = Column(String(100), nullable=True, index=True)
    prediction_type = Column(String(50), nullable=False, index=True)
    inputs = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    cache_hit = Column(Boolean, nullable=False, default=False)
    models_used = Column(JSON, nullable=True)
    missing_features = Column(JSON, nullable=True)
    method = Column(String(50), nullable=True)
    fallback_used = Column(Boolean, nullable=False, default=False)
    latency_ms = Column(Integer, nullable=True)
    status = Column(String(20), nullable=False, default="success")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


# ---------------------------------------------------------------------------
# National crop farmer guide
# ---------------------------------------------------------------------------
class CropNationalGuide(Base):
    __tablename__ = "crop_national_guides"
    __table_args__ = (
        UniqueConstraint("crop_id", name="uq_crop_national_guide_crop"),
        Index("ix_crop_national_guide_expires_at", "expires_at"),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    crop_id = Column(String(50), ForeignKey("crops.id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    generated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    provider = Column(String(50), nullable=True)
    model = Column(String(100), nullable=True)
    tokens_in = Column(Integer, nullable=True)
    tokens_out = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    version = Column(String(20), nullable=False, default="1.0")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# Municipality AI guide
# ---------------------------------------------------------------------------
class MunicipalityAIGuide(Base):
    __tablename__ = "municipality_ai_guides"
    __table_args__ = (
        UniqueConstraint("municipality_dane_code", name="uq_municipality_ai_guide"),
        Index("ix_municipality_ai_guide_expires_at", "expires_at"),
    )

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    municipality_dane_code = Column(
        String(5), ForeignKey("municipalities.dane_code"), nullable=False, index=True
    )
    content = Column(Text, nullable=False)
    generated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    provider = Column(String(50), nullable=True)
    model = Column(String(100), nullable=True)
    tokens_in = Column(Integer, nullable=True)
    tokens_out = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    version = Column(String(20), nullable=False, default="1.0")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# LLM generation audit
# ---------------------------------------------------------------------------

class LLMGeneration(Base):
    __tablename__ = "llm_generations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    provider = Column(String(50), nullable=False, index=True)
    model = Column(String(100), nullable=False)
    prompt_schema_version = Column(String(50), nullable=True)
    context_summary = Column(Text, nullable=True)
    response_json = Column(JSON, nullable=True)
    tokens_in = Column(Integer, nullable=True)
    tokens_out = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    status = Column(String(20), nullable=False, default="success")
    error_message = Column(Text, nullable=True)
    prediction_run_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
