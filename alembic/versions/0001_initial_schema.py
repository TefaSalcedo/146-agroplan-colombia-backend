"""Initial schema with normalized territorial catalog, crops, ML and audit tables

Revision ID: 0001
Revises:
Create Date: 2025-01-01 00:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- departments ---
    op.create_table(
        "departments",
        sa.Column("dane_code", sa.String(2), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # --- municipalities ---
    op.create_table(
        "municipalities",
        sa.Column("dane_code", sa.String(5), primary_key=True),
        sa.Column("department_dane_code", sa.String(2), sa.ForeignKey("departments.dane_code"), nullable=False),
        sa.Column("name", sa.String(150), nullable=False),
        sa.Column("municipality_type", sa.String(50), nullable=True),
        sa.Column("lat", sa.Float, nullable=False),
        sa.Column("lng", sa.Float, nullable=False),
        sa.Column("altitude", sa.Integer, nullable=True),
        sa.Column("avg_temperature", sa.Float, nullable=True),
        sa.Column("precipitation", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_municipalities_department_dane_code", "municipalities", ["department_dane_code"])

    # --- municipality_climate_forecasts ---
    op.create_table(
        "municipality_climate_forecasts",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("municipality_dane_code", sa.String(5), sa.ForeignKey("municipalities.dane_code"), nullable=False),
        sa.Column("forecast_date", sa.Date, nullable=False),
        sa.Column("temp_min", sa.Float, nullable=True),
        sa.Column("temp_max", sa.Float, nullable=True),
        sa.Column("temp_mean", sa.Float, nullable=True),
        sa.Column("precipitation", sa.Float, nullable=True),
        sa.Column("humidity", sa.Float, nullable=True),
        sa.Column("uv_index", sa.Float, nullable=True),
        sa.Column("wind_speed", sa.Float, nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("municipality_dane_code", "forecast_date", name="uq_climate_forecast"),
    )
    op.create_index("ix_climate_forecast_municipality", "municipality_climate_forecasts", ["municipality_dane_code"])
    op.create_index("ix_climate_forecast_date", "municipality_climate_forecasts", ["forecast_date"])

    # --- climate_sync_logs ---
    op.create_table(
        "climate_sync_logs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("sync_type", sa.String(50), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("records_processed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("records_failed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error_message", sa.Text, nullable=True),
    )

    # --- crops ---
    op.create_table(
        "crops",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("scientific_name", sa.String(200), nullable=True),
        sa.Column("image", sa.String(255), nullable=True),
        sa.Column("success_rate", sa.Integer, nullable=True),
        sa.Column("recommendation", sa.String(20), nullable=True),
        sa.Column("short_reason", sa.Text, nullable=True),
        sa.Column("reason", sa.Text, nullable=True),
        sa.Column("days_to_harvest", sa.Integer, nullable=True),
        sa.Column("soil_type", sa.String(255), nullable=True),
        sa.Column("ideal_temperature", sa.String(100), nullable=True),
        sa.Column("humidity", sa.String(100), nullable=True),
        sa.Column("precipitation", sa.String(100), nullable=True),
        sa.Column("altitude", sa.String(100), nullable=True),
        sa.Column("irrigation", sa.String(255), nullable=True),
        sa.Column("substrates", sa.JSON, nullable=True),
        sa.Column("planting_months", sa.JSON, nullable=True),
        sa.Column("harvest_months", sa.JSON, nullable=True),
        sa.Column("stages", sa.JSON, nullable=True),
        sa.Column("tips", sa.JSON, nullable=True),
        sa.Column("ml_crop_key", sa.String(50), nullable=True),
        sa.Column("is_ml_supported", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # --- model_releases ---
    op.create_table(
        "model_releases",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("model_type", sa.String(50), nullable=False),
        sa.Column("crop_key", sa.String(50), nullable=True),
        sa.Column("hf_repo", sa.String(255), nullable=False),
        sa.Column("hf_revision", sa.String(100), nullable=False),
        sa.Column("artifact_filename", sa.String(255), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=True),
        sa.Column("model_family", sa.String(50), nullable=False),
        sa.Column("preprocessor_version", sa.String(50), nullable=True),
        sa.Column("manifest", sa.JSON, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_model_releases_model_type", "model_releases", ["model_type"])
    op.create_index("ix_model_releases_crop_key", "model_releases", ["crop_key"])

    # --- prediction_cache ---
    op.create_table(
        "prediction_cache",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("cache_key", sa.String(500), nullable=False),
        sa.Column("prediction_type", sa.String(50), nullable=False),
        sa.Column("scope_key", sa.String(255), nullable=True),
        sa.Column("reference_month", sa.Date, nullable=True),
        sa.Column("algorithm_version", sa.String(100), nullable=True),
        sa.Column("payload", sa.JSON, nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("cache_key", name="uq_prediction_cache_key"),
    )
    op.create_index("ix_prediction_cache_key", "prediction_cache", ["cache_key"])
    op.create_index("ix_prediction_cache_type", "prediction_cache", ["prediction_type"])
    op.create_index("ix_prediction_cache_expires", "prediction_cache", ["expires_at"])

    # --- prediction_runs ---
    op.create_table(
        "prediction_runs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("request_id", sa.String(100), nullable=True),
        sa.Column("prediction_type", sa.String(50), nullable=False),
        sa.Column("inputs", sa.JSON, nullable=True),
        sa.Column("result", sa.JSON, nullable=True),
        sa.Column("cache_hit", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("models_used", sa.JSON, nullable=True),
        sa.Column("missing_features", sa.JSON, nullable=True),
        sa.Column("method", sa.String(50), nullable=True),
        sa.Column("fallback_used", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="success"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_prediction_runs_type", "prediction_runs", ["prediction_type"])
    op.create_index("ix_prediction_runs_request", "prediction_runs", ["request_id"])

    # --- llm_generations ---
    op.create_table(
        "llm_generations",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("prompt_schema_version", sa.String(50), nullable=True),
        sa.Column("context_summary", sa.Text, nullable=True),
        sa.Column("response_json", sa.JSON, nullable=True),
        sa.Column("tokens_in", sa.Integer, nullable=True),
        sa.Column("tokens_out", sa.Integer, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="success"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("prediction_run_id", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_llm_generations_provider", "llm_generations", ["provider"])
    op.create_index("ix_llm_generations_run", "llm_generations", ["prediction_run_id"])


def downgrade() -> None:
    op.drop_table("llm_generations")
    op.drop_table("prediction_runs")
    op.drop_table("prediction_cache")
    op.drop_table("model_releases")
    op.drop_table("crops")
    op.drop_table("climate_sync_logs")
    op.drop_table("municipality_climate_forecasts")
    op.drop_table("municipalities")
    op.drop_table("departments")
