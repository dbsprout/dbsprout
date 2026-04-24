"""Op-handler tests for the Alembic migration parser.

Covers the synthetic upgrade-body level where the parser walks each supported
Alembic op and emits typed SchemaChange records:
- TestTableOps — create_table / drop_table / rename_table.
- TestColumnAddDrop — add_column / drop_column.
- TestAlterColumn — alter_column (type, nullable, default, name).
- TestForeignKey — create_foreign_key / drop_constraint.
- TestIndex — create_index / drop_index.
"""

from __future__ import annotations

from .conftest import run_upgrade_body


class TestTableOps:
    def test_create_table_empty(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = run_upgrade_body('    op.create_table("users")')
        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.TABLE_ADDED
        assert changes[0].table_name == "users"

    def test_create_table_with_columns(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        body = (
            "    op.create_table(\n"
            '        "items",\n'
            '        sa.Column("id", sa.Integer(), nullable=False),\n'
            '        sa.Column("name", sa.String(length=120), nullable=True),\n'
            "    )"
        )
        changes = run_upgrade_body(body)
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type == SchemaChangeType.TABLE_ADDED
        assert c.table_name == "items"
        assert c.detail is not None
        cols = c.detail["columns"]
        assert cols[0]["name"] == "id"
        assert "Integer" in cols[0]["alembic_type"]
        assert cols[0]["nullable"] is False
        assert cols[1]["name"] == "name"
        assert cols[1]["nullable"] is True

    def test_drop_table(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = run_upgrade_body('    op.drop_table("users")')
        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.TABLE_REMOVED
        assert changes[0].table_name == "users"


class TestColumnAddDrop:
    def test_add_column(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        body = (
            "    op.add_column(\n"
            '        "items",\n'
            '        sa.Column("created_at", sa.DateTime(), nullable=True),\n'
            "    )"
        )
        changes = run_upgrade_body(body)
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type == SchemaChangeType.COLUMN_ADDED
        assert c.table_name == "items"
        assert c.column_name == "created_at"
        assert c.detail is not None
        assert "DateTime" in c.detail["alembic_type"]
        assert c.detail["nullable"] is True

    def test_drop_column(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = run_upgrade_body('    op.drop_column("items", "legacy_col")')
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type == SchemaChangeType.COLUMN_REMOVED
        assert c.table_name == "items"
        assert c.column_name == "legacy_col"


class TestAlterColumn:
    def test_alter_type_only(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        body = '    op.alter_column("items", "name", type_=sa.String(length=200))'
        changes = run_upgrade_body(body)
        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.COLUMN_TYPE_CHANGED
        assert changes[0].table_name == "items"
        assert changes[0].column_name == "name"
        assert changes[0].new_value is not None
        assert "String" in changes[0].new_value

    def test_alter_nullable_only(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = run_upgrade_body('    op.alter_column("items", "name", nullable=False)')
        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.COLUMN_NULLABILITY_CHANGED
        assert changes[0].new_value == "False"

    def test_alter_server_default_only(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = run_upgrade_body('    op.alter_column("items", "active", server_default="1")')
        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.COLUMN_DEFAULT_CHANGED
        assert changes[0].new_value is not None
        assert "1" in changes[0].new_value

    def test_alter_multi_axis(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        body = '    op.alter_column("items", "n", type_=sa.Text(), nullable=True)'
        changes = run_upgrade_body(body)
        kinds = {c.change_type for c in changes}
        assert kinds == {
            SchemaChangeType.COLUMN_TYPE_CHANGED,
            SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
        }

    def test_alter_no_recognized_kw_is_noop(self) -> None:
        changes = run_upgrade_body('    op.alter_column("items", "n", comment="x")')
        assert changes == []


class TestForeignKey:
    def test_create_fk(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        body = (
            "    op.create_foreign_key(\n"
            '        "fk_items_owner", "items", "users",\n'
            '        ["owner_id"], ["id"]\n'
            "    )"
        )
        changes = run_upgrade_body(body)
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type == SchemaChangeType.FOREIGN_KEY_ADDED
        assert c.table_name == "items"
        assert c.detail == {
            "name": "fk_items_owner",
            "ref_table": "users",
            "local_cols": ["owner_id"],
            "remote_cols": ["id"],
        }

    def test_drop_foreign_key_positional(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        body = '    op.drop_constraint("fk_items_owner", "items", type_="foreignkey")'
        changes = run_upgrade_body(body)
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type == SchemaChangeType.FOREIGN_KEY_REMOVED
        assert c.table_name == "items"
        assert c.detail == {"name": "fk_items_owner"}

    def test_drop_foreign_key_kwarg_table(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = run_upgrade_body(
            "    op.drop_constraint(\n"
            '        "fk_items_owner", table_name="items", type_="foreignkey"\n'
            "    )"
        )
        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.FOREIGN_KEY_REMOVED

    def test_drop_non_fk_constraint_skipped(self) -> None:
        # Only foreign-key constraints emit a change; other types are ignored.
        body = '    op.drop_constraint("uq_items_name", "items", type_="unique")'
        changes = run_upgrade_body(body)
        assert changes == []


class TestIndex:
    def test_create_index(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        body = '    op.create_index("ix_items_name", "items", ["name"], unique=True)'
        changes = run_upgrade_body(body)
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type == SchemaChangeType.INDEX_ADDED
        assert c.table_name == "items"
        assert c.detail == {
            "name": "ix_items_name",
            "cols": ["name"],
            "unique": True,
        }

    def test_create_index_default_unique_false(self) -> None:
        body = '    op.create_index("ix_items_n", "items", ["name"])'
        changes = run_upgrade_body(body)
        assert changes[0].detail is not None
        assert changes[0].detail["unique"] is False

    def test_drop_index_positional(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = run_upgrade_body('    op.drop_index("ix_items_name", "items")')
        assert changes[0].change_type == SchemaChangeType.INDEX_REMOVED
        assert changes[0].table_name == "items"
        assert changes[0].detail == {"name": "ix_items_name"}

    def test_drop_index_kwarg_table(self) -> None:
        from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

        changes = run_upgrade_body('    op.drop_index("ix_items_name", table_name="items")')
        assert changes[0].change_type == SchemaChangeType.INDEX_REMOVED
        assert changes[0].table_name == "items"
