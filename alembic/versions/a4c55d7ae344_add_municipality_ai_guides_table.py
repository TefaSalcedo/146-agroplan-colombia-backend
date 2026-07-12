"""Add municipality ai guides table

Revision ID: a4c55d7ae344
Revises: c7ed773427ee
Create Date: 2026-07-12 11:22:16.924301

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a4c55d7ae344'
down_revision: Union[str, None] = 'c7ed773427ee'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'municipality_ai_guides',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('municipality_dane_code', sa.String(length=5), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
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
        sa.ForeignKeyConstraint(['municipality_dane_code'], ['municipalities.dane_code']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('municipality_dane_code', name='uq_municipality_ai_guide')
    )
    op.create_index(
        'ix_municipality_ai_guide_expires_at',
        'municipality_ai_guides',
        ['expires_at'],
        unique=False
    )
    op.create_index(
        op.f('ix_municipality_ai_guides_municipality_dane_code'),
        'municipality_ai_guides',
        ['municipality_dane_code'],
        unique=False
    )


def downgrade() -> None:
    op.drop_index(
        op.f('ix_municipality_ai_guides_municipality_dane_code'),
        table_name='municipality_ai_guides'
    )
    op.drop_index(
        'ix_municipality_ai_guide_expires_at',
        table_name='municipality_ai_guides'
    )
    op.drop_table('municipality_ai_guides')
