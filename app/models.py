from sqlalchemy import Column, String, Float, Integer, Date, DateTime, Text
from sqlalchemy.sql import func
from app.database import Base


class Municipality(Base):
    __tablename__ = "municipalities"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    department = Column(String, nullable=False, index=True)
    lat = Column(Float, nullable=False)
    lng = Column(Float, nullable=False)
    altitude = Column(Integer, nullable=True)
    avg_temperature = Column(Float, nullable=True)
    precipitation = Column(Float, nullable=True)
    dane_code = Column(String, unique=True, index=True)


class MunicipalityClimateForecast(Base):
    __tablename__ = "municipality_climate_forecasts"

    municipality_id = Column(String, primary_key=True, index=True)
    forecast_date = Column(Date, primary_key=True, index=True)
    temp_min = Column(Float, nullable=True)
    temp_max = Column(Float, nullable=True)
    temp_mean = Column(Float, nullable=True)
    precipitation = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    uv_index = Column(Float, nullable=True)
    wind_speed = Column(Float, nullable=True)
    fetched_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class ClimateSyncLog(Base):
    __tablename__ = "climate_sync_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    sync_type = Column(String, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=False)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    records_processed = Column(Integer, nullable=False, default=0)
    records_failed = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False)
    error_message = Column(Text, nullable=True)
