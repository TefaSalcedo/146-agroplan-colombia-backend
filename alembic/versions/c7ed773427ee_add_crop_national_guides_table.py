"""Add crop national guides table

Revision ID: c7ed773427ee
Revises: d966f8ad38cc
Create Date: 2026-07-12 10:46:53.767549

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c7ed773427ee'
down_revision: Union[str, None] = 'd966f8ad38cc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'crop_national_guides',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('crop_id', sa.String(length=50), nullable=False),
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
        sa.ForeignKeyConstraint(['crop_id'], ['crops.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('crop_id', name='uq_crop_national_guide_crop')
    )
    op.create_index(
        'ix_crop_national_guide_expires_at',
        'crop_national_guides',
        ['expires_at'],
        unique=False
    )
    op.create_index(
        op.f('ix_crop_national_guides_crop_id'),
        'crop_national_guides',
        ['crop_id'],
        unique=False
    )


def downgrade() -> None:
    op.drop_index(
        op.f('ix_crop_national_guides_crop_id'),
        table_name='crop_national_guides'
    )
    op.drop_index(
        'ix_crop_national_guide_expires_at',
        table_name='crop_national_guides'
    )
    op.drop_table('crop_national_guides')
