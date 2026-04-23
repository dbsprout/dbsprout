"""init"""

revision = "0001"
down_revision = None

import sqlalchemy as sa
from alembic import op


def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)


def downgrade():
    pass
