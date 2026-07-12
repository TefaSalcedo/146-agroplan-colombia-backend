"""add_municipality_current_weather_table

Revision ID: b3abe1832336
Revises: 0001
Create Date: 2026-07-12 01:46:24.326199

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3abe1832336'
down_revision: Union[str, None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'municipality_current_weather',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('municipality_dane_code', sa.String(length=5), nullable=False),
        sa.Column('temperature', sa.Float(), nullable=False),
        sa.Column('condition', sa.String(length=100), nullable=False),
        sa.Column('humidity', sa.Float(), nullable=False),
        sa.Column('precipitation', sa.Float(), nullable=False),
        sa.Column('icon', sa.String(length=50), nullable=False),
        sa.Column('source', sa.String(length=50), nullable=False),
        sa.Column('fetched_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['municipality_dane_code'], ['municipalities.dane_code']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('municipality_dane_code', name='uq_current_weather_municipality')
    )
    op.create_index(op.f('ix_municipality_current_weather_id'), 'municipality_current_weather', ['id'], unique=False)
    op.create_index(op.f('ix_municipality_current_weather_municipality_dane_code'), 'municipality_current_weather', ['municipality_dane_code'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_municipality_current_weather_municipality_dane_code'), table_name='municipality_current_weather')
    op.drop_index(op.f('ix_municipality_current_weather_id'), table_name='municipality_current_weather')
    op.drop_table('municipality_current_weather')
