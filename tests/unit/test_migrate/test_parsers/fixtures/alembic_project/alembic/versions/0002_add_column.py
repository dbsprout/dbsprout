"""add column"""

revision = "0002"
down_revision = "0001"

import sqlalchemy as sa
from alembic import op


def upgrade():
    op.add_column("users", sa.Column("created_at", sa.DateTime(), nullable=True))
    op.alter_column("users", "email", type_=sa.String(length=320), nullable=False)


def downgrade():
    pass
