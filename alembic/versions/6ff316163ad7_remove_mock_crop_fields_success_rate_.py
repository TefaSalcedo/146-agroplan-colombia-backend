"""remove mock crop fields success_rate recommendation short_reason reason

Revision ID: 6ff316163ad7
Revises: b3abe1832336
Create Date: 2026-07-12 02:46:02.532632

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6ff316163ad7'
down_revision: Union[str, None] = 'b3abe1832336'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column('crops', 'recommendation')
    op.drop_column('crops', 'short_reason')
    op.drop_column('crops', 'reason')
    op.drop_column('crops', 'success_rate')


def downgrade() -> None:
    op.add_column('crops', sa.Column('success_rate', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('crops', sa.Column('recommendation', sa.VARCHAR(length=20), autoincrement=False, nullable=True))
    op.add_column('crops', sa.Column('short_reason', sa.TEXT(), autoincrement=False, nullable=True))
    op.add_column('crops', sa.Column('reason', sa.TEXT(), autoincrement=False, nullable=True))
