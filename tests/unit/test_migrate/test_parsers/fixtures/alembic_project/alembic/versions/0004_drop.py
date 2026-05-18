"""drop"""

revision = "0004"
down_revision = "0003"

from alembic import op


def upgrade():
    op.drop_constraint("fk_posts_user", "posts", type_="foreignkey")
    op.drop_index("ix_users_email", "users")
    op.drop_column("users", "created_at")
    op.drop_table("posts")


def downgrade():
    pass
