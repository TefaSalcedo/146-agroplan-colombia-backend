"""Add crop establishment period and perennial flag

Revision ID: 76161805c979
Revises: 561f363290de
Create Date: 2026-07-12 12:08:06.204103

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '76161805c979'
down_revision: Union[str, None] = '561f363290de'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('crops', sa.Column('establishment_period_days', sa.Integer(), nullable=True))
    op.add_column('crops', sa.Column('is_perennial', sa.Boolean(), nullable=False, server_default=sa.text('false')))


def downgrade() -> None:
    op.drop_column('crops', 'is_perennial')
    op.drop_column('crops', 'establishment_period_days')
