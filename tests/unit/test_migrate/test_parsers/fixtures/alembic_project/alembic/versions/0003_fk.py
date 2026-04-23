"""fk"""

revision = "0003"
down_revision = "0002"

import sqlalchemy as sa
from alembic import op


def upgrade():
    op.create_table(
        "posts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
    )
    op.create_foreign_key("fk_posts_user", "posts", "users", ["user_id"], ["id"])


def downgrade():
    pass
