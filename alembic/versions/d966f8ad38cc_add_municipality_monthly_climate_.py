"""Add municipality monthly climate forecast table

Revision ID: d966f8ad38cc
Revises: 6ff316163ad7
Create Date: 2026-07-12 08:18:01.064123

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd966f8ad38cc'
down_revision: Union[str, None] = '6ff316163ad7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'municipality_monthly_climate_forecasts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('municipality_dane_code', sa.String(length=5), nullable=False),
        sa.Column('forecast_month', sa.Date(), nullable=False),
        sa.Column('temp_mean', sa.Float(), nullable=True),
        sa.Column('temp_anomaly', sa.Float(), nullable=True),
        sa.Column('precipitation', sa.Float(), nullable=True),
        sa.Column('precipitation_anomaly', sa.Float(), nullable=True),
        sa.Column('trend', sa.String(length=50), nullable=True),
        sa.Column('source', sa.String(length=50), nullable=False),
        sa.Column('fetched_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['municipality_dane_code'], ['municipalities.dane_code']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('municipality_dane_code', 'forecast_month', name='uq_monthly_climate_forecast')
    )
    op.create_index(
        op.f('ix_municipality_monthly_climate_forecasts_forecast_month'),
        'municipality_monthly_climate_forecasts',
        ['forecast_month'],
        unique=False
    )
    op.create_index(
        op.f('ix_municipality_monthly_climate_forecasts_municipality_dane_code'),
        'municipality_monthly_climate_forecasts',
        ['municipality_dane_code'],
        unique=False
    )


def downgrade() -> None:
    op.drop_index(
        op.f('ix_municipality_monthly_climate_forecasts_municipality_dane_code'),
        table_name='municipality_monthly_climate_forecasts'
    )
    op.drop_index(
        op.f('ix_municipality_monthly_climate_forecasts_forecast_month'),
        table_name='municipality_monthly_climate_forecasts'
    )
    op.drop_table('municipality_monthly_climate_forecasts')
