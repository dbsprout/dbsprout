from __future__ import annotations

from pathlib import Path

from dbsprout.migrate.parsers.alembic import AlembicParser


class TestIntegrationFixture:
    FIXTURE = Path(__file__).parent / "fixtures" / "alembic_project"

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
