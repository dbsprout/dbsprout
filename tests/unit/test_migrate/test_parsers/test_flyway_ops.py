# tests/unit/test_migrate/test_parsers/test_flyway_ops.py
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import sqlglot
from sqlglot import exp

from dbsprout.migrate.models import SchemaChange, SchemaChangeType
from dbsprout.migrate.parsers import MigrationParseError, MigrationParser
from dbsprout.migrate.parsers.flyway import (
    FlywayMigrationParser,
    _discover_migration_files,
    _extract_inline_fks,
    _FKLedger,
    _handle_alter_column,
    _handle_create_index,
    _handle_create_table,
    _parse_file,
    _split_qualified,
    _strip_quotes,
    _walk_statements,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestScaffold:
    def test_implements_protocol(self) -> None:
        assert isinstance(FlywayMigrationParser(), MigrationParser)

    def test_empty_project_raises(self, tmp_path: Path) -> None:
        with pytest.raises(MigrationParseError, match=r"no V\*__\*\.sql found"):
            FlywayMigrationParser().detect_changes(tmp_path)

    def test_frozen_dataclass(self) -> None:
        parser = FlywayMigrationParser()
        with pytest.raises(dataclasses.FrozenInstanceError):
            parser.dialect = "mysql"  # type: ignore[misc]

    def test_default_dialect_is_postgres(self) -> None:
        assert FlywayMigrationParser().dialect == "postgres"


class TestIdentifierNorm:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('"Users"', "Users"),
            ("`Users`", "Users"),
            ("[Users]", "Users"),
            ("users", "users"),
        ],
    )
    def test_strip_quotes(self, raw: str, expected: str) -> None:
        assert _strip_quotes(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("users", (None, "users")),
            ("public.users", ("public", "users")),
            ('"app"."orders"', ("app", "orders")),
        ],
    )
    def test_split_qualified(self, raw: str, expected: tuple[str | None, str]) -> None:
        assert _split_qualified(raw) == expected


class TestParseFile:
    def test_parse_error_wrapped(self, tmp_path: Path) -> None:
        file_path = tmp_path / "V1__bad.sql"
        file_path.write_text("THIS IS NOT SQL ${{{;;;", encoding="utf-8")
        with pytest.raises(MigrationParseError, match="could not parse"):
            _parse_file(file_path, dialect="postgres", placeholders={})

    def test_placeholder_applied_before_parse(self, tmp_path: Path) -> None:
        file_path = tmp_path / "V1__ok.sql"
        file_path.write_text("CREATE TABLE ${schema}.users (id INT);", encoding="utf-8")
        stmts = _parse_file(file_path, dialect="postgres", placeholders={"schema": "public"})
        assert len(stmts) == 1

    def test_unresolved_placeholder_raises(self, tmp_path: Path) -> None:
        file_path = tmp_path / "V1__ok.sql"
        file_path.write_text("CREATE TABLE ${schema}.users (id INT);", encoding="utf-8")
        with pytest.raises(MigrationParseError, match="unresolved placeholder"):
            _parse_file(file_path, dialect="postgres", placeholders={})


class TestCreateDropTable:
    def test_create_table(self) -> None:
        stmts = sqlglot.parse(
            "CREATE TABLE authors (id INT PRIMARY KEY, name VARCHAR(120) NOT NULL);",
            read="postgres",
        )
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type is SchemaChangeType.TABLE_ADDED
        assert c.table_name == "authors"
        assert c.detail is not None
        cols = c.detail["columns"]
        assert cols[0]["name"] == "id"
        assert cols[0]["primary_key"] is True
        assert cols[1]["name"] == "name"
        assert cols[1]["sql_type"].upper().startswith("VARCHAR")
        assert cols[1]["nullable"] is False

    def test_create_table_with_inline_fk(self) -> None:
        stmts = sqlglot.parse(
            "CREATE TABLE books (id INT, author_id INT REFERENCES authors(id));",
            read="postgres",
        )
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert changes[0].change_type is SchemaChangeType.TABLE_ADDED
        fks = changes[0].detail["foreign_keys"]
        assert fks[0]["ref_table"] == "authors"
        assert fks[0]["local_cols"] == ["author_id"]
        assert fks[0]["remote_cols"] == ["id"]

    def test_create_table_with_schema_prefix(self) -> None:
        stmts = sqlglot.parse("CREATE TABLE app.orders (id INT);", read="postgres")
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert changes[0].table_name == "orders"
        assert changes[0].detail["schema"] == "app"

    def test_drop_table(self) -> None:
        stmts = sqlglot.parse("DROP TABLE authors;", read="postgres")
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert len(changes) == 1
        assert changes[0].change_type is SchemaChangeType.TABLE_REMOVED
        assert changes[0].table_name == "authors"


class TestAddDropColumn:
    def test_add_column(self) -> None:
        stmts = sqlglot.parse(
            "ALTER TABLE users ADD COLUMN email VARCHAR(254) NOT NULL DEFAULT '';",
            read="postgres",
        )
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type is SchemaChangeType.COLUMN_ADDED
        assert c.table_name == "users"
        assert c.column_name == "email"
        assert c.detail["sql_type"].upper().startswith("VARCHAR")
        assert c.detail["nullable"] is False
        assert c.detail["default"] is not None

    def test_add_column_with_inline_fk(self) -> None:
        stmts = sqlglot.parse(
            "ALTER TABLE books ADD COLUMN author_id INT REFERENCES authors(id);",
            read="postgres",
        )
        changes = _walk_statements(stmts, ledger=_FKLedger())
        # Two changes: COLUMN_ADDED + FOREIGN_KEY_ADDED
        kinds = [c.change_type for c in changes]
        assert SchemaChangeType.COLUMN_ADDED in kinds
        assert SchemaChangeType.FOREIGN_KEY_ADDED in kinds
        fk = next(c for c in changes if c.change_type is SchemaChangeType.FOREIGN_KEY_ADDED)
        assert fk.detail["ref_table"] == "authors"
        assert fk.detail["local_cols"] == ["author_id"]

    def test_drop_column(self) -> None:
        stmts = sqlglot.parse("ALTER TABLE users DROP COLUMN legacy_flag;", read="postgres")
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type is SchemaChangeType.COLUMN_REMOVED
        assert c.table_name == "users"
        assert c.column_name == "legacy_flag"


class TestAlterColumn:
    def test_alter_type(self) -> None:
        stmts = sqlglot.parse("ALTER TABLE t ALTER COLUMN c TYPE BIGINT;", read="postgres")
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type is SchemaChangeType.COLUMN_TYPE_CHANGED
        assert c.column_name == "c"
        assert c.new_value is not None
        assert c.new_value.upper() == "BIGINT"

    def test_alter_set_not_null(self) -> None:
        stmts = sqlglot.parse("ALTER TABLE t ALTER COLUMN c SET NOT NULL;", read="postgres")
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert changes[0].change_type is SchemaChangeType.COLUMN_NULLABILITY_CHANGED
        assert changes[0].new_value == "NOT NULL"

    def test_alter_drop_not_null(self) -> None:
        stmts = sqlglot.parse("ALTER TABLE t ALTER COLUMN c DROP NOT NULL;", read="postgres")
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert changes[0].change_type is SchemaChangeType.COLUMN_NULLABILITY_CHANGED
        assert changes[0].new_value == "NULL"

    def test_alter_set_default(self) -> None:
        stmts = sqlglot.parse(
            "ALTER TABLE t ALTER COLUMN c SET DEFAULT 42;",
            read="postgres",
        )
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert changes[0].change_type is SchemaChangeType.COLUMN_DEFAULT_CHANGED
        assert changes[0].new_value == "42"

    def test_alter_drop_default(self) -> None:
        stmts = sqlglot.parse(
            "ALTER TABLE t ALTER COLUMN c DROP DEFAULT;",
            read="postgres",
        )
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert changes[0].change_type is SchemaChangeType.COLUMN_DEFAULT_CHANGED
        assert changes[0].new_value is None


class TestAddDropConstraint:
    def test_add_fk_constraint(self) -> None:
        fk_add_sql = (
            "ALTER TABLE books ADD CONSTRAINT fk_author"
            " FOREIGN KEY (author_id) REFERENCES authors(id);"
        )
        stmts = sqlglot.parse(fk_add_sql, read="postgres")
        ledger = _FKLedger()
        changes = _walk_statements(stmts, ledger=ledger)
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type is SchemaChangeType.FOREIGN_KEY_ADDED
        assert c.table_name == "books"
        assert c.detail["constraint_name"] == "fk_author"
        assert c.detail["ref_table"] == "authors"
        assert ledger.resolve("books", "fk_author") is not None

    def test_drop_fk_resolved_by_ledger(self) -> None:
        fk_add_sql = (
            "ALTER TABLE books ADD CONSTRAINT fk_author"
            " FOREIGN KEY (author_id) REFERENCES authors(id);"
        )
        add = sqlglot.parse(fk_add_sql, read="postgres")
        drop = sqlglot.parse(
            "ALTER TABLE books DROP CONSTRAINT fk_author;",
            read="postgres",
        )
        ledger = _FKLedger()
        _walk_statements(add, ledger=ledger)
        changes = _walk_statements(drop, ledger=ledger)
        assert len(changes) == 1
        assert changes[0].change_type is SchemaChangeType.FOREIGN_KEY_REMOVED
        assert changes[0].detail["constraint_name"] == "fk_author"

    def test_drop_unknown_constraint_skipped(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        stmts = sqlglot.parse(
            "ALTER TABLE books DROP CONSTRAINT chk_price;",
            read="postgres",
        )
        with caplog.at_level("DEBUG", logger="dbsprout.migrate.parsers"):
            changes = _walk_statements(stmts, ledger=_FKLedger())
        assert changes == []
        assert "chk_price" in caplog.text


class TestCreateDropIndex:
    def test_create_index(self) -> None:
        stmts = sqlglot.parse(
            "CREATE INDEX idx_users_email ON users (email);",
            read="postgres",
        )
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type is SchemaChangeType.INDEX_ADDED
        assert c.table_name == "users"
        assert c.detail["index_name"] == "idx_users_email"
        assert c.detail["cols"] == ["email"]

    def test_drop_index(self) -> None:
        stmts = sqlglot.parse("DROP INDEX idx_users_email;", read="postgres")
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert len(changes) == 1
        c = changes[0]
        assert c.change_type is SchemaChangeType.INDEX_REMOVED
        assert c.detail["index_name"] == "idx_users_email"
        # DROP INDEX in Postgres carries no table name. Contract: emit ""
        # as the sentinel. Downstream adapters reconcile via detail["index_name"].
        assert c.table_name == ""


class TestRename:
    def test_rename_table_emits_pair(self) -> None:
        stmts = sqlglot.parse("ALTER TABLE old_name RENAME TO new_name;", read="postgres")
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert len(changes) == 2
        rm, add = changes
        assert rm.change_type is SchemaChangeType.TABLE_REMOVED
        assert rm.table_name == "old_name"
        assert add.change_type is SchemaChangeType.TABLE_ADDED
        assert add.table_name == "new_name"
        assert add.detail["rename_of"] == "old_name"
        assert rm.detail["rename_of"] == "new_name"

    def test_rename_column_emits_pair(self) -> None:
        stmts = sqlglot.parse(
            "ALTER TABLE users RENAME COLUMN uname TO username;",
            read="postgres",
        )
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert len(changes) == 2
        rm, add = changes
        assert rm.change_type is SchemaChangeType.COLUMN_REMOVED
        assert rm.column_name == "uname"
        assert add.change_type is SchemaChangeType.COLUMN_ADDED
        assert add.column_name == "username"
        assert add.detail["rename_of"] == "uname"
        assert rm.detail["rename_of"] == "username"


class TestEdgeCases:
    """Cover defensive branches that normal SQL doesn't exercise."""

    def test_split_qualified_empty_returns_none_empty(self) -> None:
        assert _split_qualified("") == (None, "")

    def test_fk_ledger_record_no_name_is_noop(self) -> None:
        ledger = _FKLedger()
        change = SchemaChange(
            change_type=SchemaChangeType.FOREIGN_KEY_ADDED,
            table_name="t",
            detail={"constraint_name": None, "local_cols": [], "ref_table": "", "remote_cols": []},
        )
        ledger.record(change)
        assert ledger.by_key == {}

    def test_create_table_as_select_skipped(self) -> None:
        stmts = sqlglot.parse("CREATE TABLE t AS SELECT 1;", read="postgres")
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert changes == []

    def test_handle_create_table_non_schema_returns_empty(self) -> None:
        node = sqlglot.parse("CREATE TABLE t AS SELECT 1;", read="postgres")[0]
        assert isinstance(node, exp.Create)
        assert _handle_create_table(node) == []

    def test_drop_table_with_schema_prefix(self) -> None:
        stmts = sqlglot.parse("DROP TABLE app.orders;", read="postgres")
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert len(changes) == 1
        assert changes[0].table_name == "orders"
        assert changes[0].detail is not None
        assert changes[0].detail.get("schema") == "app"

    def test_extract_inline_fks_non_schema_returns_empty(self) -> None:
        node = sqlglot.parse("CREATE TABLE t AS SELECT 1;", read="postgres")[0]
        assert isinstance(node, exp.Create)
        assert _extract_inline_fks(node) == []

    def test_add_column_with_schema_prefix(self) -> None:
        stmts = sqlglot.parse("ALTER TABLE app.users ADD COLUMN notes TEXT;", read="postgres")
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert len(changes) == 1
        assert changes[0].change_type is SchemaChangeType.COLUMN_ADDED
        assert changes[0].detail.get("schema") == "app"

    def test_alter_column_no_dimension_returns_empty(self) -> None:
        action = exp.AlterColumn(this=exp.Column(this=exp.Identifier(this="c")))
        changes = _handle_alter_column("t", action)
        assert changes == []

    def test_add_constraint_non_fk_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        stmts = sqlglot.parse("ALTER TABLE t ADD CONSTRAINT uq_col UNIQUE (col);", read="postgres")
        with caplog.at_level("DEBUG", logger="dbsprout.migrate.parsers"):
            changes = _walk_statements(stmts, ledger=_FKLedger())
        assert changes == []
        assert "not a FK" in caplog.text

    def test_alter_table_unsupported_action_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        stmts = sqlglot.parse("ALTER TABLE t SET TABLESPACE tblspc;", read="postgres")
        with caplog.at_level("DEBUG", logger="dbsprout.migrate.parsers"):
            changes = _walk_statements(stmts, ledger=_FKLedger())
        assert changes == []
        assert "AlterSet" in caplog.text

    def test_alter_table_drop_index_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        stmts = sqlglot.parse("ALTER TABLE t DROP INDEX my_idx;", read="mysql")
        with caplog.at_level("DEBUG", logger="dbsprout.migrate.parsers"):
            changes = _walk_statements(stmts, ledger=_FKLedger())
        assert changes == []
        assert "unsupported ALTER TABLE DROP action" in caplog.text

    def test_unsupported_statement_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        stmts = sqlglot.parse("INSERT INTO t VALUES (1);", read="postgres")
        with caplog.at_level("DEBUG", logger="dbsprout.migrate.parsers"):
            changes = _walk_statements(stmts, ledger=_FKLedger())
        assert changes == []
        assert "skipping unsupported statement" in caplog.text

    def test_handle_create_index_non_index_returns_empty(self) -> None:
        node = sqlglot.parse("CREATE TABLE t (id INT);", read="postgres")[0]
        assert isinstance(node, exp.Create)
        assert _handle_create_index(node) == []

    def test_discovery_oserror_on_stat_skips_file(self, tmp_path: Path) -> None:
        from tests.unit.test_migrate.test_parsers.conftest import (  # noqa: PLC0415
            build_flyway_project,
        )

        build_flyway_project(tmp_path, {"V1__ok": "CREATE TABLE t (id INT);"})
        build_flyway_project(tmp_path, {"V2__bad": "-- bad file"})
        from pathlib import Path as _Path  # noqa: PLC0415

        bad_file = tmp_path / "db" / "migration" / "V2__bad.sql"
        orig_stat = _Path.stat

        def _selective_oserror(self: _Path, **kwargs: object) -> object:  # type: ignore[misc]
            if self == bad_file:
                raise OSError("no access")
            return orig_stat(self, **kwargs)  # type: ignore[call-arg]

        with patch.object(_Path, "stat", _selective_oserror):
            files = _discover_migration_files(tmp_path, None)
        assert len(files) == 1
        assert files[0].name == "V1__ok.sql"

    def test_discovery_non_flyway_sql_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from tests.unit.test_migrate.test_parsers.conftest import (  # noqa: PLC0415
            build_flyway_project,
        )

        build_flyway_project(tmp_path, {"V1__ok": "-- ok"})
        (tmp_path / "db" / "migration" / "helpers.sql").write_text("-- helper", encoding="utf-8")
        with caplog.at_level("DEBUG", logger="dbsprout.migrate.parsers"):
            files = _discover_migration_files(tmp_path, None)
        assert len(files) == 1
        assert "helpers.sql" in caplog.text

    def test_extract_inline_fks_column_ref_no_schema_wrapper(self) -> None:
        """MySQL REFERENCES without column list → ref is exp.Table, not exp.Schema."""
        stmts = sqlglot.parse(
            "CREATE TABLE books (id INT, author_id INT REFERENCES authors);",
            read="mysql",
        )
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert changes[0].change_type is SchemaChangeType.TABLE_ADDED
        fks = changes[0].detail["foreign_keys"]
        assert fks[0]["ref_table"] == "authors"

    def test_add_column_inline_fk_no_schema_wrapper(self) -> None:
        """MySQL ADD COLUMN REFERENCES authors (no column list) → non-Schema ref path."""
        stmts = sqlglot.parse(
            "ALTER TABLE books ADD COLUMN author_id INT REFERENCES authors;",
            read="mysql",
        )
        changes = _walk_statements(stmts, ledger=_FKLedger())
        kinds = [c.change_type for c in changes]
        assert SchemaChangeType.COLUMN_ADDED in kinds
        assert SchemaChangeType.FOREIGN_KEY_ADDED in kinds
        fk = next(c for c in changes if c.change_type is SchemaChangeType.FOREIGN_KEY_ADDED)
        assert fk.detail["ref_table"] == "authors"

    def test_add_fk_constraint_no_schema_wrapper(self) -> None:
        """MySQL ADD CONSTRAINT FK with ref as Table (no column list) → ref_this Table path."""
        fk_sql = (
            "ALTER TABLE books ADD CONSTRAINT fk_author FOREIGN KEY (author_id) REFERENCES authors;"
        )
        stmts = sqlglot.parse(fk_sql, read="mysql")
        ledger = _FKLedger()
        changes = _walk_statements(stmts, ledger=ledger)
        assert len(changes) == 1
        assert changes[0].change_type is SchemaChangeType.FOREIGN_KEY_ADDED

    def test_add_fk_constraint_no_constraint_name(self) -> None:
        """MySQL ADD FOREIGN KEY without constraint name → ForeignKey directly in expressions."""
        fk_sql = "ALTER TABLE books ADD FOREIGN KEY (author_id) REFERENCES authors(id);"
        stmts = sqlglot.parse(fk_sql, read="mysql")
        ledger = _FKLedger()
        changes = _walk_statements(stmts, ledger=ledger)
        assert len(changes) == 1
        assert changes[0].change_type is SchemaChangeType.FOREIGN_KEY_ADDED

    def test_create_index_with_expression_column(self) -> None:
        """CREATE INDEX with functional expression (lower(col)) → non-Column inner node."""
        stmts = sqlglot.parse(
            "CREATE INDEX idx_lower ON t (lower(name));",
            read="postgres",
        )
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert len(changes) == 1
        assert changes[0].change_type is SchemaChangeType.INDEX_ADDED

    def test_discovery_custom_location_nonexistent_dir_skipped(self, tmp_path: Path) -> None:
        """Custom location that doesn't exist → dir skipped (hits not d.is_dir() branch)."""
        from tests.unit.test_migrate.test_parsers.conftest import (  # noqa: PLC0415
            build_flyway_project,
        )

        build_flyway_project(tmp_path, {"V1__ok": "CREATE TABLE t (id INT);"}, location="sql/v1")
        files = _discover_migration_files(tmp_path, ("sql/v1", "sql/missing"))
        assert len(files) == 1
        assert files[0].name == "V1__ok.sql"

    def test_create_table_with_table_level_fk_no_col_list(self) -> None:
        """Table-level FOREIGN KEY REFERENCES without column list → fk_ref.this is Table."""
        sql = (
            "CREATE TABLE books ("
            "  id INT,"
            "  author_id INT,"
            "  FOREIGN KEY (author_id) REFERENCES authors"
            ");"
        )
        stmts = sqlglot.parse(sql, read="mysql")
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert changes[0].change_type is SchemaChangeType.TABLE_ADDED
        fks = changes[0].detail["foreign_keys"]
        assert fks[0]["ref_table"] == "authors"

    def test_add_constraint_fk_ref_is_table(self) -> None:
        """ADD CONSTRAINT FK where ref.this is Table (MySQL no column list)."""
        fk_sql = (
            "ALTER TABLE books ADD CONSTRAINT fk_author FOREIGN KEY (author_id) REFERENCES authors;"
        )
        stmts = sqlglot.parse(fk_sql, read="mysql")
        ledger = _FKLedger()
        changes = _walk_statements(stmts, ledger=ledger)
        assert len(changes) == 1
        assert changes[0].change_type is SchemaChangeType.FOREIGN_KEY_ADDED
        assert changes[0].detail["ref_table"] == "authors"

    def test_create_table_with_mixed_table_level_constraints(self) -> None:
        """CREATE TABLE with PRIMARY KEY + FK at table level → PK expr skipped."""
        sql = (
            "CREATE TABLE books ("
            "  id INT,"
            "  author_id INT,"
            "  PRIMARY KEY (id),"
            "  FOREIGN KEY (author_id) REFERENCES authors(id)"
            ");"
        )
        stmts = sqlglot.parse(sql, read="postgres")
        changes = _walk_statements(stmts, ledger=_FKLedger())
        assert changes[0].change_type is SchemaChangeType.TABLE_ADDED
        fks = changes[0].detail["foreign_keys"]
        assert len(fks) == 1
        assert fks[0]["local_cols"] == ["author_id"]
