"""Add crop municipality recommendations cache table

Revision ID: 561f363290de
Revises: e2e0625cd2e3
Create Date: 2026-07-12 12:00:03.045193

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '561f363290de'
down_revision: Union[str, None] = 'e2e0625cd2e3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'crop_municipality_recommendations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('crop_id', sa.String(length=50), nullable=False),
        sa.Column('municipality_dane_code', sa.String(length=5), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('generated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('provider', sa.String(length=50), nullable=True),
        sa.Column('model', sa.String(length=100), nullable=True),
        sa.Column('tokens_in', sa.Integer(), nullable=True),
        sa.Column('tokens_out', sa.Integer(), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('version', sa.String(length=20), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['crop_id'], ['crops.id']),
        sa.ForeignKeyConstraint(['municipality_dane_code'], ['municipalities.dane_code']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('crop_id', 'municipality_dane_code', name='uq_crop_municipality_recommendation')
    )
    op.create_index(
        'ix_crop_municipality_recommendation_expires_at',
        'crop_municipality_recommendations',
        ['expires_at'],
        unique=False
    )


def downgrade() -> None:
    op.drop_index(
        'ix_crop_municipality_recommendation_expires_at',
        table_name='crop_municipality_recommendations'
    )
    op.drop_table('crop_municipality_recommendations')
