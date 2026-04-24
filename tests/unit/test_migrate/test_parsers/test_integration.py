"""End-to-end golden-fixture test for the Alembic migration parser.

Drives AlembicParser().detect_changes() against the full fixtures/alembic_project/
tree and asserts the exact change_type + table_name sequence.

Lives in its own module (rather than test_alembic_*) so future per-framework
parsers (Django, Flyway, Liquibase, Prisma) can add sibling TestIntegrationFixture
classes here instead of each growing their own monolith file.
"""

from __future__ import annotations

from dbsprout.migrate.parsers.alembic import AlembicParser

from .conftest import FIXTURE_PROJECT_PATH


class TestIntegrationFixture:
    FIXTURE = FIXTURE_PROJECT_PATH

    def test_detect_changes_golden(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = AlembicParser().detect_changes(self.FIXTURE)
        kinds = [c.change_type for c in changes]
        tables = [c.table_name for c in changes]

        # 0001: create_table(users) + create_index(ix_users_email)
        # 0002: add_column(users.created_at) + alter_column type_+nullable -> 2 changes
        # 0003: create_table(posts) + create_foreign_key
        # 0004: drop_constraint fk + drop_index + drop_column + drop_table
        assert kinds == [
            SchemaChangeType.TABLE_ADDED,
            SchemaChangeType.INDEX_ADDED,
            SchemaChangeType.COLUMN_ADDED,
            SchemaChangeType.COLUMN_TYPE_CHANGED,
            SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
            SchemaChangeType.TABLE_ADDED,
            SchemaChangeType.FOREIGN_KEY_ADDED,
            SchemaChangeType.FOREIGN_KEY_REMOVED,
            SchemaChangeType.INDEX_REMOVED,
            SchemaChangeType.COLUMN_REMOVED,
            SchemaChangeType.TABLE_REMOVED,
        ]
        assert tables == [
            "users",
            "users",
            "users",
            "users",
            "users",
            "posts",
            "posts",
            "posts",
            "users",
            "users",
            "posts",
        ]
