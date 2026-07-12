"""Add municipality climate enrichment table

Revision ID: e2e0625cd2e3
Revises: a4c55d7ae344
Create Date: 2026-07-12 11:38:28.582626

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e2e0625cd2e3'
down_revision: Union[str, None] = 'a4c55d7ae344'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'municipality_climate_enrichment',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('municipality_dane_code', sa.String(length=5), nullable=False),
        sa.Column('altitude', sa.Integer(), nullable=True),
        sa.Column('avg_temperature', sa.Float(), nullable=True),
        sa.Column('precipitation', sa.Float(), nullable=True),
        sa.Column('source', sa.String(length=50), nullable=False),
        sa.Column('fetched_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['municipality_dane_code'], ['municipalities.dane_code']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('municipality_dane_code', name='uq_municipality_climate_enrichment')
    )
    op.create_index(
        op.f('ix_municipality_climate_enrichment_municipality_dane_code'),
        'municipality_climate_enrichment',
        ['municipality_dane_code'],
        unique=False
    )


def downgrade() -> None:
    op.drop_index(
        op.f('ix_municipality_climate_enrichment_municipality_dane_code'),
        table_name='municipality_climate_enrichment'
    )
    op.drop_table('municipality_climate_enrichment')
