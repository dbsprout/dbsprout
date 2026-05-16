"""Tests for dbsprout diff CLI command."""

from __future__ import annotations

import json
import re
import string
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import sqlalchemy as sa
from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.cli.commands.diff import _change_prefix, _summarize
from dbsprout.migrate.models import SchemaChange, SchemaChangeType
from dbsprout.migrate.snapshot import SnapshotStore
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    IndexSchema,
    TableSchema,
)
from dbsprout.schema.parsers.ddl import parse_ddl

if TYPE_CHECKING:
    from pathlib import Path

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes for assertion matching."""
    return _ANSI_RE.sub("", text)


def _simple_schema_for_diff() -> DatabaseSchema:
    """Minimal schema for diff tests."""
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="users",
                columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)],
                primary_key=["id"],
            )
        ],
        dialect="sqlite",
    )


class TestDiffHelp:
    def test_help_shows_usage(self) -> None:
        result = runner.invoke(app, ["diff", "--help"])
        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "--db" in output
        assert "--file" in output
        assert "--snapshot" in output
        assert "--format" in output
        assert "--output-dir" in output
        assert "Report schema changes" in output


class TestDiffArgValidation:
    def test_db_and_file_both_provided_exits_2(self) -> None:
        result = runner.invoke(app, ["diff", "--db", "sqlite:///x", "--file", "schema.sql"])
        assert result.exit_code == 2
        output = _strip_ansi(result.output)
        assert "only one of --db or --file" in output.lower()

    def test_invalid_format_exits_2(self) -> None:
        result = runner.invoke(app, ["diff", "--db", "sqlite:///x", "--format", "xml"])
        assert result.exit_code == 2
        output = _strip_ansi(result.output)
        assert "invalid format" in output.lower()


class TestDiffErrorPaths:
    def test_no_source_exits_2(self, tmp_path: Path) -> None:
        """No --db, no --file, no config.schema.source → exit 2 with actionable error."""
        result = runner.invoke(app, ["diff", "--output-dir", str(tmp_path)])
        assert result.exit_code == 2
        output = _strip_ansi(result.output)
        assert "no schema source" in output.lower()

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    def test_no_snapshots_exits_2(self, mock_store_cls: MagicMock, tmp_path: Path) -> None:
        """--db provided but .dbsprout/snapshots/ is empty → exit 2."""
        mock_store = MagicMock()
        mock_store.load_latest.return_value = None
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            ["diff", "--db", "sqlite:///x.db", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 2
        output = _strip_ansi(result.output)
        assert "no snapshots found" in output.lower()


class TestDiffSnapshotFlag:
    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    def test_snapshot_hash_not_found_exits_2(
        self, mock_store_cls: MagicMock, tmp_path: Path
    ) -> None:
        """--snapshot HASH that matches no snapshot file → exit 2."""
        mock_store = MagicMock()
        mock_store.load_by_hash.return_value = None
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            [
                "diff",
                "--db",
                "sqlite:///x.db",
                "--snapshot",
                "deadbeef",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 2
        output = _strip_ansi(result.output)
        assert "snapshot not found" in output.lower()
        assert "deadbeef" in output
        mock_store.load_latest.assert_not_called()
        mock_store.load_by_hash.assert_called_once_with("deadbeef")

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    def test_snapshot_hash_found_falls_through_to_next_stage(
        self, mock_store_cls: MagicMock, tmp_path: Path
    ) -> None:
        """--snapshot HASH that matches → load_by_hash returns schema, falls through."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)],
                    primary_key=["id"],
                )
            ],
            dialect="sqlite",
        )
        mock_store = MagicMock()
        mock_store.load_by_hash.return_value = schema
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            [
                "diff",
                "--db",
                "sqlite:///x.db",
                "--snapshot",
                "abc12345",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code != 2
        mock_store.load_by_hash.assert_called_once_with("abc12345")
        mock_store.load_latest.assert_not_called()


class TestDiffDbIntrospection:
    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_db_url_introspects_live_schema(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--db URL triggers introspect() call."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)],
                    primary_key=["id"],
                )
            ],
            dialect="sqlite",
        )
        mock_introspect.return_value = schema
        mock_store = MagicMock()
        mock_store.load_latest.return_value = schema
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            ["diff", "--db", "sqlite:///x.db", "--output-dir", str(tmp_path)],
        )
        # Falls through to NotImplementedError — exit != 2 means introspection succeeded
        assert result.exit_code != 2
        mock_introspect.assert_called_once_with("sqlite:///x.db")

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_db_introspection_failure_exits_2(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """SQLAlchemyError during introspect → exit 2."""
        mock_introspect.side_effect = sa.exc.OperationalError(
            "connect", {}, Exception("connection refused")
        )
        mock_store = MagicMock()
        mock_store.load_latest.return_value = _simple_schema_for_diff()
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            ["diff", "--db", "sqlite:///bad.db", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 2
        output = _strip_ansi(result.output)
        assert "error" in output.lower()

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_db_password_not_leaked_on_introspection_error(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A bad URL with a password must NOT leak the password in the error output."""
        mock_introspect.side_effect = sa.exc.OperationalError(
            "connect", {}, Exception("auth failed for postgresql://user:supersecret@host/db")
        )
        mock_store = MagicMock()
        mock_store.load_latest.return_value = _simple_schema_for_diff()
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            [
                "diff",
                "--db",
                "postgresql://user:supersecret@host/db",
                "--format",
                "json",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 2
        assert "supersecret" not in _strip_ansi(result.output), (
            "DB password leaked in diff command output via exception message"
        )

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_db_password_not_leaked_rich_format(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Rich (default) format must also scrub the password from error output."""
        mock_introspect.side_effect = sa.exc.OperationalError(
            "connect", {}, Exception("auth failed for postgresql://user:supersecret@host/db")
        )
        mock_store = MagicMock()
        mock_store.load_latest.return_value = _simple_schema_for_diff()
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            [
                "diff",
                "--db",
                "postgresql://user:supersecret@host/db",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 2
        assert "supersecret" not in _strip_ansi(result.output), (
            "DB password leaked in diff command Rich output via exception message"
        )


class TestDiffFilePath:
    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    def test_file_not_found_exits_2(self, mock_store_cls: MagicMock, tmp_path: Path) -> None:
        """--file pointing at nonexistent path → exit 2."""
        mock_store = MagicMock()
        mock_store.load_latest.return_value = _simple_schema_for_diff()
        mock_store_cls.return_value = mock_store

        missing = tmp_path / "missing.sql"
        result = runner.invoke(
            app,
            ["diff", "--file", str(missing), "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 2
        output = _strip_ansi(result.output)
        assert "file not found" in output.lower()
        assert str(missing) in output

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    def test_file_sql_ddl_parsed(self, mock_store_cls: MagicMock, tmp_path: Path) -> None:
        """--file with valid SQL DDL parses and flows past source resolution."""
        sql_file = tmp_path / "schema.sql"
        sql_file.write_text("CREATE TABLE users (id INTEGER PRIMARY KEY, email VARCHAR(255));")
        mock_store = MagicMock()
        mock_store.load_latest.return_value = _simple_schema_for_diff()
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            ["diff", "--file", str(sql_file), "--output-dir", str(tmp_path)],
        )
        # NotImplementedError at end of pipeline → exit != 2 means file parsing succeeded
        assert result.exit_code != 2

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    def test_file_parse_error_exits_2(self, mock_store_cls: MagicMock, tmp_path: Path) -> None:
        """--file with unparseable SQL → exit 2."""
        bad_file = tmp_path / "broken.sql"
        bad_file.write_text("THIS IS NOT VALID SQL DDL @@@ garbage")
        mock_store = MagicMock()
        mock_store.load_latest.return_value = _simple_schema_for_diff()
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            ["diff", "--file", str(bad_file), "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 2
        output = _strip_ansi(result.output)
        assert "error" in output.lower()


class TestDiffConfigFallback:
    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_falls_back_to_config_db_url(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """No flags → reads dbsprout.toml → uses schema.source as DB URL."""
        toml_content = (
            "[schema]\n"
            'dialect = "sqlite"\n'
            'source = "sqlite:///./data.db"\n'
            'snapshot = ".dbsprout/snapshots/"\n'
        )
        (tmp_path / "dbsprout.toml").write_text(toml_content)

        mock_introspect.return_value = _simple_schema_for_diff()
        mock_store = MagicMock()
        mock_store.load_latest.return_value = _simple_schema_for_diff()
        mock_store_cls.return_value = mock_store

        result = runner.invoke(app, ["diff", "--output-dir", str(tmp_path)])
        assert result.exit_code != 2
        mock_introspect.assert_called_once_with("sqlite:///./data.db")

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    def test_falls_back_to_config_file_path(
        self, mock_store_cls: MagicMock, tmp_path: Path
    ) -> None:
        """No flags → reads dbsprout.toml → treats schema.source as file path."""
        sql_file = tmp_path / "schema.sql"
        sql_file.write_text("CREATE TABLE users (id INTEGER PRIMARY KEY, name VARCHAR(255));")
        toml_content = f'[schema]\ndialect = "sqlite"\nsource = "{sql_file}"\n'
        (tmp_path / "dbsprout.toml").write_text(toml_content)

        mock_store = MagicMock()
        mock_store.load_latest.return_value = _simple_schema_for_diff()
        mock_store_cls.return_value = mock_store

        result = runner.invoke(app, ["diff", "--output-dir", str(tmp_path)])
        assert result.exit_code != 2

    def test_config_exists_but_source_is_empty_exits_2(self, tmp_path: Path) -> None:
        """dbsprout.toml exists but schema.source is missing or empty → exit 2."""
        toml_content = '[schema]\ndialect = "sqlite"\n# source not set\n'
        (tmp_path / "dbsprout.toml").write_text(toml_content)

        result = runner.invoke(app, ["diff", "--output-dir", str(tmp_path)])
        assert result.exit_code == 2
        output = _strip_ansi(result.output)
        assert "no schema source" in output.lower()


class TestDiffNoChanges:
    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_no_changes_exits_0_rich_format(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Identical old and new schemas → exit 0, 'No changes detected' printed."""
        schema = _simple_schema_for_diff()
        mock_introspect.return_value = schema
        mock_store = MagicMock()
        mock_store.load_latest.return_value = schema
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            ["diff", "--db", "sqlite:///x.db", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "no changes detected" in output.lower()

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_no_changes_exits_0_json_format(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Identical schemas with --format json → exit 0, valid JSON payload."""
        schema = _simple_schema_for_diff()
        mock_introspect.return_value = schema
        mock_store = MagicMock()
        mock_store.load_latest.return_value = schema
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            [
                "diff",
                "--db",
                "sqlite:///x.db",
                "--format",
                "json",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["summary"]["total"] == 0
        assert payload["changes"] == []
        assert "old_snapshot" in payload
        assert "new_source" in payload
        assert "generated_at" in payload


class TestDiffSummaryHelper:
    def test_summarize_empty_list(self) -> None:
        """Empty change list → total 0, every change type 0."""
        result = _summarize([])
        assert result["total"] == 0
        for ct in SchemaChangeType:
            assert result[ct.value] == 0

    def test_summarize_covers_all_change_types(self) -> None:
        """One SchemaChange per SchemaChangeType → total == count, each type == 1."""
        changes = [
            SchemaChange(change_type=ct, table_name=f"t_{ct.value}") for ct in SchemaChangeType
        ]
        result = _summarize(changes)

        assert result["total"] == len(list(SchemaChangeType))
        for ct in SchemaChangeType:
            assert result[ct.value] == 1


class TestDiffRichRender:
    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_rich_panel_header_and_table_added(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """TABLE_ADDED change → Panel header + Tables section."""
        old = _simple_schema_for_diff()
        new = DatabaseSchema(
            tables=[
                *old.tables,
                TableSchema(
                    name="orders",
                    columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)],
                    primary_key=["id"],
                ),
            ],
            dialect="sqlite",
        )
        mock_introspect.return_value = new
        mock_store = MagicMock()
        mock_store.load_latest.return_value = old
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            ["diff", "--db", "sqlite:///x.db", "--output-dir", str(tmp_path)],
        )
        output = _strip_ansi(result.output)
        # Panel header elements
        assert "Schema Drift" in output
        assert "old:" in output
        assert "new:" in output
        assert "summary:" in output
        assert "generated:" in output
        assert "+tables: 1" in output
        # Grouped section
        assert "Tables" in output
        assert "+ orders" in output
        assert result.exit_code == 1  # drift detected → CI signal

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_rich_column_type_change_format(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """COLUMN_TYPE_CHANGED → shows `table.col: old_type → new_type`."""
        old = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                        ColumnSchema(name="email", data_type=ColumnType.VARCHAR, max_length=255),
                    ],
                    primary_key=["id"],
                )
            ],
            dialect="sqlite",
        )
        new = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                        ColumnSchema(name="email", data_type=ColumnType.TEXT),
                    ],
                    primary_key=["id"],
                )
            ],
            dialect="sqlite",
        )
        mock_introspect.return_value = new
        mock_store = MagicMock()
        mock_store.load_latest.return_value = old
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            ["diff", "--db", "sqlite:///x.db", "--output-dir", str(tmp_path)],
        )
        output = _strip_ansi(result.output)
        assert "Columns" in output
        assert "users.email" in output
        assert "→" in output
        # Should NOT show sentinel
        assert "__enums__" not in output
        assert result.exit_code == 1

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_rich_enum_changed_uses_name_not_sentinel(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """ENUM_CHANGED → renders under 'Enums' with the enum name, never '__enums__'."""
        old = DatabaseSchema(
            tables=[
                TableSchema(
                    name="orders",
                    columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)],
                    primary_key=["id"],
                )
            ],
            enums={"order_status": ["pending", "shipped"]},
            dialect="postgresql",
        )
        new = DatabaseSchema(
            tables=[
                TableSchema(
                    name="orders",
                    columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)],
                    primary_key=["id"],
                )
            ],
            enums={"order_status": ["pending", "shipped", "cancelled"]},
            dialect="postgresql",
        )
        mock_introspect.return_value = new
        mock_store = MagicMock()
        mock_store.load_latest.return_value = old
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            ["diff", "--db", "postgresql://host/db", "--output-dir", str(tmp_path)],
        )
        output = _strip_ansi(result.output)
        assert "Enums" in output
        assert "order_status" in output
        assert "enum: order_status" in output
        assert "__enums__" not in output
        assert result.exit_code == 1


class TestDiffJsonRender:
    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_json_shape_and_content(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--format json on non-empty diff → valid JSON with expected shape."""
        old = _simple_schema_for_diff()
        new = DatabaseSchema(
            tables=[
                *old.tables,
                TableSchema(
                    name="orders",
                    columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)],
                    primary_key=["id"],
                ),
            ],
            dialect="sqlite",
        )
        mock_introspect.return_value = new
        mock_store = MagicMock()
        mock_store.load_latest.return_value = old
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            [
                "diff",
                "--db",
                "sqlite:///x.db",
                "--format",
                "json",
                "--output-dir",
                str(tmp_path),
            ],
        )
        payload = json.loads(result.output)
        assert payload["summary"]["total"] == 1
        assert payload["summary"]["table_added"] == 1
        assert len(payload["changes"]) == 1
        assert payload["changes"][0]["change_type"] == "table_added"
        assert "old_snapshot" in payload
        assert "new_source" in payload
        assert "generated_at" in payload

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_json_output_has_no_ansi_codes(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """JSON output must be parseable — no Rich markup leaks."""
        old = _simple_schema_for_diff()
        new = DatabaseSchema(
            tables=[
                *old.tables,
                TableSchema(
                    name="orders",
                    columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)],
                    primary_key=["id"],
                ),
            ],
            dialect="sqlite",
        )
        mock_introspect.return_value = new
        mock_store = MagicMock()
        mock_store.load_latest.return_value = old
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            [
                "diff",
                "--db",
                "sqlite:///x.db",
                "--format",
                "json",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert "\x1b[" not in result.output
        # And it must parse as valid JSON
        json.loads(result.output)


class TestDiffExitCodes:
    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_exit_1_on_drift_rich(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Non-empty diff in rich format → exit 1 (drift signal for CI)."""
        old = _simple_schema_for_diff()
        new = DatabaseSchema(
            tables=[
                *old.tables,
                TableSchema(
                    name="orders",
                    columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)],
                    primary_key=["id"],
                ),
            ],
            dialect="sqlite",
        )
        mock_introspect.return_value = new
        mock_store = MagicMock()
        mock_store.load_latest.return_value = old
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            ["diff", "--db", "sqlite:///x.db", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 1

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_exit_1_on_drift_json(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Non-empty diff in json format → exit 1."""
        old = _simple_schema_for_diff()
        new = DatabaseSchema(
            tables=[
                *old.tables,
                TableSchema(
                    name="orders",
                    columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)],
                    primary_key=["id"],
                ),
            ],
            dialect="sqlite",
        )
        mock_introspect.return_value = new
        mock_store = MagicMock()
        mock_store.load_latest.return_value = old
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            [
                "diff",
                "--db",
                "sqlite:///x.db",
                "--format",
                "json",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 1


class TestDiffSecurity:
    """Regression coverage that the render path sanitizes DB passwords."""

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_password_sanitized_in_json_output(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """JSON output's new_source must NOT contain the password from --db."""
        old = _simple_schema_for_diff()
        new = DatabaseSchema(
            tables=[
                *old.tables,
                TableSchema(
                    name="orders",
                    columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)],
                    primary_key=["id"],
                ),
            ],
            dialect="postgresql",
        )
        mock_introspect.return_value = new
        mock_store = MagicMock()
        mock_store.load_latest.return_value = old
        mock_store_cls.return_value = mock_store

        secret = "supersecret123"  # noqa: S105 — test fixture only
        url = f"postgresql://myuser:{secret}@db.example.com:5432/mydb"

        result = runner.invoke(
            app,
            ["diff", "--db", url, "--format", "json", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 1  # drift detected
        assert secret not in result.output
        payload = json.loads(result.output)
        assert secret not in payload["new_source"]
        # The sanitized URL should still contain the username and host
        assert "myuser" in payload["new_source"]
        assert "db.example.com" in payload["new_source"]

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_password_sanitized_in_rich_output(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Rich Panel's 'new:' line must NOT contain the password from --db."""
        old = _simple_schema_for_diff()
        new = DatabaseSchema(
            tables=[
                *old.tables,
                TableSchema(
                    name="orders",
                    columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)],
                    primary_key=["id"],
                ),
            ],
            dialect="postgresql",
        )
        mock_introspect.return_value = new
        mock_store = MagicMock()
        mock_store.load_latest.return_value = old
        mock_store_cls.return_value = mock_store

        secret = "topsecret_password!"  # noqa: S105 — test fixture only
        url = f"postgresql://alice:{secret}@prod.example.com/app"

        result = runner.invoke(
            app,
            ["diff", "--db", url, "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 1
        output = _strip_ansi(result.output)
        assert secret not in output
        assert "alice" in output  # username is fine
        assert "prod.example.com" in output  # host is fine


class TestDiffCorruptSnapshots:
    """Verify diff survives corrupt snapshot files (SnapshotStore drops them)."""

    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_corrupt_latest_snapshot_falls_back_to_older(
        self,
        mock_introspect: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A corrupt newest snapshot + a valid older one → diff uses the older, succeeds.

        This is an integration test that does NOT mock SnapshotStore — it uses
        a real store over a real filesystem.
        """
        snap_dir = tmp_path / ".dbsprout" / "snapshots"
        snap_dir.mkdir(parents=True)

        # Valid older snapshot via the real store (timestamp earlier)
        valid_schema = _simple_schema_for_diff()
        store = SnapshotStore(base_dir=snap_dir)
        store.save(valid_schema)

        # Find the written file and verify it exists
        written = list(snap_dir.glob("*.json"))
        assert len(written) == 1

        # Corrupt NEWER snapshot (lexicographically later filename)
        corrupt_path = snap_dir / "99999999T999999Z_deadbeef.json"
        corrupt_path.write_text("this is not valid json {{{")

        # Introspect returns a schema with one extra table → drift expected
        new_schema = DatabaseSchema(
            tables=[
                *valid_schema.tables,
                TableSchema(
                    name="orders",
                    columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)],
                    primary_key=["id"],
                ),
            ],
            dialect="sqlite",
        )
        mock_introspect.return_value = new_schema

        result = runner.invoke(
            app,
            ["diff", "--db", "sqlite:///x.db", "--output-dir", str(tmp_path)],
        )
        # The corrupt file should be skipped, older valid one used → diff works
        assert result.exit_code == 1  # drift detected against valid snapshot
        output = _strip_ansi(result.output)
        assert "orders" in output

    def test_all_snapshots_corrupt_exits_2(self, tmp_path: Path) -> None:
        """Every snapshot file is corrupt → SnapshotStore returns None → exit 2."""
        snap_dir = tmp_path / ".dbsprout" / "snapshots"
        snap_dir.mkdir(parents=True)

        (snap_dir / "20260101T000000Z_aaaaaaaa.json").write_text("garbage 1")
        (snap_dir / "20260102T000000Z_bbbbbbbb.json").write_text("garbage 2")

        result = runner.invoke(
            app,
            ["diff", "--db", "sqlite:///x.db", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 2
        output = _strip_ansi(result.output)
        assert "no snapshots found" in output.lower()


class TestDiffCoverageGaps:
    """Tests targeting specific uncovered branches in diff.py."""

    def test_malformed_toml_exits_2(self, tmp_path: Path) -> None:
        """Malformed dbsprout.toml → load_config raises ValueError → exit 2."""
        (tmp_path / "dbsprout.toml").write_text("[schema\nthis is not valid toml ====")
        result = runner.invoke(app, ["diff", "--output-dir", str(tmp_path)])
        assert result.exit_code == 2
        output = _strip_ansi(result.output)
        assert "no schema source" in output.lower()

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_rich_renders_foreign_key_changes(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """FK_REMOVED should appear in the 'Foreign Keys' group."""
        old = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)],
                    primary_key=["id"],
                ),
                TableSchema(
                    name="orders",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                        ColumnSchema(
                            name="user_id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                        ),
                    ],
                    primary_key=["id"],
                    foreign_keys=[
                        ForeignKeySchema(
                            columns=["user_id"],
                            ref_table="users",
                            ref_columns=["id"],
                        ),
                    ],
                ),
            ],
            dialect="sqlite",
        )
        new = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)],
                    primary_key=["id"],
                ),
                TableSchema(
                    name="orders",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                        ColumnSchema(
                            name="user_id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                        ),
                    ],
                    primary_key=["id"],
                    foreign_keys=[],
                ),
            ],
            dialect="sqlite",
        )
        mock_introspect.return_value = new
        mock_store = MagicMock()
        mock_store.load_latest.return_value = old
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app, ["diff", "--db", "sqlite:///x.db", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code == 1
        output = _strip_ansi(result.output)
        assert "Foreign Keys" in output

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_rich_renders_index_changes(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """INDEX_ADDED should appear in the 'Indexes' group."""
        old = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                        ColumnSchema(
                            name="email",
                            data_type=ColumnType.VARCHAR,
                            max_length=255,
                        ),
                    ],
                    primary_key=["id"],
                    indexes=[],
                )
            ],
            dialect="sqlite",
        )
        new = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                        ColumnSchema(
                            name="email",
                            data_type=ColumnType.VARCHAR,
                            max_length=255,
                        ),
                    ],
                    primary_key=["id"],
                    indexes=[IndexSchema(name="idx_email", columns=["email"], unique=True)],
                )
            ],
            dialect="sqlite",
        )
        mock_introspect.return_value = new
        mock_store = MagicMock()
        mock_store.load_latest.return_value = old
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app, ["diff", "--db", "sqlite:///x.db", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code == 1
        output = _strip_ansi(result.output)
        assert "Indexes" in output

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    def test_file_dbml_parsed(self, mock_store_cls: MagicMock, tmp_path: Path) -> None:
        """--file with .dbml suffix → dispatches to parse_dbml."""
        dbml_file = tmp_path / "schema.dbml"
        dbml_file.write_text(
            "Table users {\n"
            "  id integer [pk]\n"
            "  email varchar [not null, unique]\n"
            "  name varchar\n"
            "}\n"
        )
        mock_store = MagicMock()
        mock_store.load_latest.return_value = _simple_schema_for_diff()
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app, ["diff", "--file", str(dbml_file), "--output-dir", str(tmp_path)]
        )
        assert result.exit_code != 2

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    def test_file_mermaid_parsed(self, mock_store_cls: MagicMock, tmp_path: Path) -> None:
        """--file with .mermaid suffix → dispatches to parse_mermaid."""
        mm_file = tmp_path / "schema.mermaid"
        mm_file.write_text(
            "erDiagram\n"
            "    USERS {\n"
            "        int id PK\n"
            "        string email\n"
            "        string name\n"
            "    }\n"
        )
        mock_store = MagicMock()
        mock_store.load_latest.return_value = _simple_schema_for_diff()
        mock_store_cls.return_value = mock_store

        result = runner.invoke(app, ["diff", "--file", str(mm_file), "--output-dir", str(tmp_path)])
        assert result.exit_code != 2

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    def test_file_plantuml_parsed(self, mock_store_cls: MagicMock, tmp_path: Path) -> None:
        """--file with .puml suffix → dispatches to parse_plantuml."""
        puml_file = tmp_path / "schema.puml"
        puml_file.write_text(
            "@startuml\n"
            'entity "users" as users {\n'
            "  *id : integer <<PK>>\n"
            "  --\n"
            "  *email : varchar\n"
            "  name : varchar\n"
            "}\n"
            "@enduml\n"
        )
        mock_store = MagicMock()
        mock_store.load_latest.return_value = _simple_schema_for_diff()
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app, ["diff", "--file", str(puml_file), "--output-dir", str(tmp_path)]
        )
        assert result.exit_code != 2

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    def test_file_prisma_parsed(self, mock_store_cls: MagicMock, tmp_path: Path) -> None:
        """--file with .prisma suffix → dispatches to parse_prisma."""
        prisma_file = tmp_path / "schema.prisma"
        prisma_file.write_text(
            "model User {\n"
            "  id    Int    @id @default(autoincrement())\n"
            "  email String @unique\n"
            "  name  String?\n"
            "}\n"
        )
        mock_store = MagicMock()
        mock_store.load_latest.return_value = _simple_schema_for_diff()
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app, ["diff", "--file", str(prisma_file), "--output-dir", str(tmp_path)]
        )
        assert result.exit_code != 2


class TestDiffStartup:
    """AC-29: diff must be in --help and the CLI app must not eager-import sqlalchemy."""

    def test_diff_in_root_help(self) -> None:
        """``dbsprout --help`` must list the diff command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "diff" in _strip_ansi(result.output)

    def test_cli_app_does_not_eagerly_import_sqlalchemy(self) -> None:
        """Importing ``dbsprout.cli.app`` must not pull in sqlalchemy (cold-start target).

        We save and restore ``sys.modules`` to avoid polluting other tests that
        rely on class identity for cached dbsprout modules.
        """
        import importlib  # noqa: PLC0415 — test-local, after snapshotting sys.modules
        import sys  # noqa: PLC0415

        prefixes = ("dbsprout", "sqlalchemy")
        saved = {name: mod for name, mod in sys.modules.items() if name.startswith(prefixes)}
        try:
            for name in list(sys.modules):
                if name.startswith(prefixes):
                    sys.modules.pop(name, None)

            importlib.import_module("dbsprout.cli.app")

            assert "sqlalchemy" not in sys.modules, (
                "dbsprout.cli.app pulls in sqlalchemy at import time — "
                "diff_proxy must lazy-import diff_command"
            )
        finally:
            for name in list(sys.modules):
                if name.startswith(prefixes):
                    sys.modules.pop(name, None)
            sys.modules.update(saved)


class TestDiffChangePrefix:
    """AC-10: direct unit tests on the colour-coded ``_change_prefix`` helper."""

    def test_added_prefix_is_green(self) -> None:
        assert _change_prefix(SchemaChangeType.TABLE_ADDED) == "[green]+[/green]"
        assert _change_prefix(SchemaChangeType.COLUMN_ADDED) == "[green]+[/green]"
        assert _change_prefix(SchemaChangeType.FOREIGN_KEY_ADDED) == "[green]+[/green]"
        assert _change_prefix(SchemaChangeType.INDEX_ADDED) == "[green]+[/green]"

    def test_removed_prefix_is_red(self) -> None:
        assert _change_prefix(SchemaChangeType.TABLE_REMOVED) == "[red]-[/red]"
        assert _change_prefix(SchemaChangeType.COLUMN_REMOVED) == "[red]-[/red]"
        assert _change_prefix(SchemaChangeType.FOREIGN_KEY_REMOVED) == "[red]-[/red]"
        assert _change_prefix(SchemaChangeType.INDEX_REMOVED) == "[red]-[/red]"

    def test_changed_prefix_is_yellow(self) -> None:
        assert _change_prefix(SchemaChangeType.COLUMN_TYPE_CHANGED) == "[yellow]~[/yellow]"
        assert _change_prefix(SchemaChangeType.COLUMN_NULLABILITY_CHANGED) == "[yellow]~[/yellow]"
        assert _change_prefix(SchemaChangeType.COLUMN_DEFAULT_CHANGED) == "[yellow]~[/yellow]"
        assert _change_prefix(SchemaChangeType.ENUM_CHANGED) == "[yellow]~[/yellow]"


class TestDiffColumnMetaChanges:
    """AC-11: column nullability and default render tests."""

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_rich_column_nullability_change_format(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """COLUMN_NULLABILITY_CHANGED → shows ``table.col: nullable OLD → NEW``."""
        old = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                        ColumnSchema(
                            name="email",
                            data_type=ColumnType.VARCHAR,
                            max_length=255,
                            nullable=True,
                        ),
                    ],
                    primary_key=["id"],
                )
            ],
            dialect="sqlite",
        )
        new = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                        ColumnSchema(
                            name="email",
                            data_type=ColumnType.VARCHAR,
                            max_length=255,
                            nullable=False,
                        ),
                    ],
                    primary_key=["id"],
                )
            ],
            dialect="sqlite",
        )
        mock_introspect.return_value = new
        mock_store = MagicMock()
        mock_store.load_latest.return_value = old
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            ["diff", "--db", "sqlite:///x.db", "--output-dir", str(tmp_path)],
        )
        output = _strip_ansi(result.output)
        assert result.exit_code == 1
        assert "users.email: nullable" in output
        assert "→" in output

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_rich_column_default_change_format(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """COLUMN_DEFAULT_CHANGED → shows ``table.col: default OLD → NEW``."""
        old = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                        ColumnSchema(
                            name="email",
                            data_type=ColumnType.VARCHAR,
                            max_length=255,
                            default="'old@example.com'",
                        ),
                    ],
                    primary_key=["id"],
                )
            ],
            dialect="sqlite",
        )
        new = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                        ColumnSchema(
                            name="email",
                            data_type=ColumnType.VARCHAR,
                            max_length=255,
                            default="'new@example.com'",
                        ),
                    ],
                    primary_key=["id"],
                )
            ],
            dialect="sqlite",
        )
        mock_introspect.return_value = new
        mock_store = MagicMock()
        mock_store.load_latest.return_value = old
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            ["diff", "--db", "sqlite:///x.db", "--output-dir", str(tmp_path)],
        )
        output = _strip_ansi(result.output)
        assert result.exit_code == 1
        assert "users.email: default" in output
        assert "→" in output


class TestDiffGroupOrdering:
    """AC-12: grouped change sections render in the canonical order."""

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_rich_groups_render_in_canonical_order(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Tables appear before Columns before Foreign Keys before Indexes before Enums."""
        old = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                    ],
                    primary_key=["id"],
                ),
                TableSchema(
                    name="orders",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                        ColumnSchema(
                            name="user_id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                        ),
                    ],
                    primary_key=["id"],
                    foreign_keys=[
                        ForeignKeySchema(
                            columns=["user_id"],
                            ref_table="users",
                            ref_columns=["id"],
                        ),
                    ],
                ),
                TableSchema(
                    name="products",
                    columns=[
                        ColumnSchema(name="sku", data_type=ColumnType.VARCHAR, max_length=64),
                    ],
                    primary_key=["sku"],
                    indexes=[IndexSchema(name="idx_sku", columns=["sku"], unique=True)],
                ),
            ],
            enums={"status": ["active", "inactive"]},
            dialect="sqlite",
        )
        new = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                        ColumnSchema(
                            name="email",
                            data_type=ColumnType.VARCHAR,
                            max_length=255,
                        ),
                    ],
                    primary_key=["id"],
                ),
                TableSchema(
                    name="orders",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                        ColumnSchema(
                            name="user_id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                        ),
                    ],
                    primary_key=["id"],
                    foreign_keys=[],
                ),
                TableSchema(
                    name="products",
                    columns=[
                        ColumnSchema(name="sku", data_type=ColumnType.VARCHAR, max_length=64),
                    ],
                    primary_key=["sku"],
                    indexes=[],
                ),
                TableSchema(
                    name="categories",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                    ],
                    primary_key=["id"],
                ),
            ],
            enums={"status": ["active", "inactive", "pending"]},
            dialect="sqlite",
        )
        mock_introspect.return_value = new
        mock_store = MagicMock()
        mock_store.load_latest.return_value = old
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            ["diff", "--db", "sqlite:///x.db", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 1
        output = _strip_ansi(result.output)

        tables_pos = output.index("\nTables")
        columns_pos = output.index("\nColumns")
        fks_pos = output.index("\nForeign Keys")
        indexes_pos = output.index("\nIndexes")
        enums_pos = output.index("\nEnums")

        assert tables_pos < columns_pos < fks_pos < indexes_pos < enums_pos


class TestDiffFileSourceMatrix:
    """File-source x format matrix coverage (Reviewer 4)."""

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    def test_rich_file_source_no_changes(
        self,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """File source + identical schemas → exit 0, Rich 'No changes detected'."""
        sql_file = tmp_path / "schema.sql"
        sql_file.write_text("CREATE TABLE users (id INTEGER PRIMARY KEY);")

        parsed = parse_ddl(sql_file.read_text(), source_file=str(sql_file))

        mock_store = MagicMock()
        mock_store.load_latest.return_value = parsed
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            ["diff", "--file", str(sql_file), "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "no changes detected" in output.lower()

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    def test_rich_file_source_drift(
        self,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """File source + drift → exit 1, Rich panel shows drift."""
        sql_file = tmp_path / "schema.sql"
        sql_file.write_text(
            "CREATE TABLE users (id INTEGER PRIMARY KEY);\n"
            "CREATE TABLE orders (id INTEGER PRIMARY KEY);"
        )

        mock_store = MagicMock()
        mock_store.load_latest.return_value = _simple_schema_for_diff()
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            ["diff", "--file", str(sql_file), "--output-dir", str(tmp_path)],
        )
        assert result.exit_code == 1
        output = _strip_ansi(result.output)
        assert "Schema Drift" in output
        assert "+tables: 1" in output

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    def test_json_file_source_no_changes(
        self,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """File source + identical → exit 0, valid JSON with total == 0."""
        sql_file = tmp_path / "schema.sql"
        sql_file.write_text("CREATE TABLE users (id INTEGER PRIMARY KEY);")

        parsed = parse_ddl(sql_file.read_text(), source_file=str(sql_file))

        mock_store = MagicMock()
        mock_store.load_latest.return_value = parsed
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            [
                "diff",
                "--file",
                str(sql_file),
                "--format",
                "json",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["summary"]["total"] == 0
        assert payload["changes"] == []
        assert payload["new_source"] == str(sql_file)

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    def test_json_file_source_drift(
        self,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """File source + drift → exit 1, valid JSON with drift."""
        sql_file = tmp_path / "schema.sql"
        sql_file.write_text(
            "CREATE TABLE users (id INTEGER PRIMARY KEY);\n"
            "CREATE TABLE orders (id INTEGER PRIMARY KEY);"
        )

        mock_store = MagicMock()
        mock_store.load_latest.return_value = _simple_schema_for_diff()
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            [
                "diff",
                "--file",
                str(sql_file),
                "--format",
                "json",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["summary"]["total"] >= 1
        assert payload["summary"]["table_added"] == 1
        assert payload["new_source"] == str(sql_file)


class TestDiffJsonFieldFormats:
    """JSON field format assertions (Reviewer 4)."""

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_json_generated_at_is_iso8601_utc(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """``generated_at`` must be ISO 8601 format and UTC timezone."""
        old = _simple_schema_for_diff()
        new = DatabaseSchema(
            tables=[
                *old.tables,
                TableSchema(
                    name="orders",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="sqlite",
        )
        mock_introspect.return_value = new
        mock_store = MagicMock()
        mock_store.load_latest.return_value = old
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            [
                "diff",
                "--db",
                "sqlite:///x.db",
                "--format",
                "json",
                "--output-dir",
                str(tmp_path),
            ],
        )
        payload = json.loads(result.output)
        parsed = datetime.fromisoformat(payload["generated_at"])
        assert parsed.tzinfo is not None, "generated_at must include timezone"
        assert parsed.utcoffset() == timezone.utc.utcoffset(parsed), "generated_at must be UTC"

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    @patch("dbsprout.cli.commands.diff._introspect_db")
    def test_json_old_snapshot_is_8_char_hex_prefix(
        self,
        mock_introspect: MagicMock,
        mock_store_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """``old_snapshot`` should be exactly 8 characters (schema_hash prefix)."""
        old = _simple_schema_for_diff()
        new = DatabaseSchema(
            tables=[
                *old.tables,
                TableSchema(
                    name="orders",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="sqlite",
        )
        mock_introspect.return_value = new
        mock_store = MagicMock()
        mock_store.load_latest.return_value = old
        mock_store_cls.return_value = mock_store

        result = runner.invoke(
            app,
            [
                "diff",
                "--db",
                "sqlite:///x.db",
                "--format",
                "json",
                "--output-dir",
                str(tmp_path),
            ],
        )
        payload = json.loads(result.output)
        assert len(payload["old_snapshot"]) == 8
        assert all(c in string.hexdigits for c in payload["old_snapshot"])


class TestDiffSchemaChangeTypeGrouping:
    """Canary: every SchemaChangeType variant must be grouped by _render_rich_changes."""

    def test_all_schema_change_types_are_grouped(self) -> None:
        """If a new variant is added, this test fails with a clear message."""
        table_group = {SchemaChangeType.TABLE_ADDED, SchemaChangeType.TABLE_REMOVED}
        column_group = {
            SchemaChangeType.COLUMN_ADDED,
            SchemaChangeType.COLUMN_REMOVED,
            SchemaChangeType.COLUMN_TYPE_CHANGED,
            SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
            SchemaChangeType.COLUMN_DEFAULT_CHANGED,
        }
        fk_group = {
            SchemaChangeType.FOREIGN_KEY_ADDED,
            SchemaChangeType.FOREIGN_KEY_REMOVED,
        }
        index_group = {
            SchemaChangeType.INDEX_ADDED,
            SchemaChangeType.INDEX_REMOVED,
        }
        enum_group = {SchemaChangeType.ENUM_CHANGED}

        all_grouped = table_group | column_group | fk_group | index_group | enum_group
        missing = set(SchemaChangeType) - all_grouped

        assert not missing, (
            f"SchemaChangeType variants not grouped in _render_rich_changes: {missing}. "
            "Update dbsprout/cli/commands/diff.py::_render_rich_changes to group these."
        )


# ── S-054a security hardening (cohesive block; sibling S-054b layers on top) ──


class TestDiffHashValidation:
    """S-054a AC-4/AC-6: --snapshot hex validation + escaped not-found message."""

    def test_non_hex_snapshot_rejected_by_callback(self, tmp_path: Path) -> None:
        """A non-hex --snapshot prefix is rejected by the Typer callback
        (usage error), not the generic 'not found' fallback."""
        result = runner.invoke(
            app,
            [
                "diff",
                "--db",
                "sqlite:///x.db",
                "--snapshot",
                "zz//../etc",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 2
        out = _strip_ansi(result.output)
        assert "Invalid value for '--snapshot'" in out
        assert "hex" in out.lower()
        # It must be rejected BEFORE the not-found fallback runs.
        assert "snapshot not found" not in out.lower()

    def test_uppercase_hex_snapshot_rejected_by_callback(self, tmp_path: Path) -> None:
        """Uppercase hex is rejected by the callback (filenames use lowercase)."""
        result = runner.invoke(
            app,
            [
                "diff",
                "--db",
                "sqlite:///x.db",
                "--snapshot",
                "DEADBEEF",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 2
        out = _strip_ansi(result.output)
        assert "Invalid value for '--snapshot'" in out

    def test_valid_lowercase_hex_snapshot_passes_callback(self, tmp_path: Path) -> None:
        """A valid lowercase-hex prefix passes the callback (no BadParameter)."""
        result = runner.invoke(
            app,
            [
                "diff",
                "--db",
                "sqlite:///x.db",
                "--snapshot",
                "deadbeef",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert "Invalid value for '--snapshot'" not in _strip_ansi(result.output)

    @patch("dbsprout.migrate.snapshot.SnapshotStore")
    def test_snapshot_not_found_message_is_markup_escaped(
        self, mock_store_cls: MagicMock, tmp_path: Path
    ) -> None:
        """AC-6: the echoed hash is markup-escaped (no-op for hex, but the
        escape() call is exercised and the literal hash is shown)."""
        mock_store = MagicMock()
        mock_store.load_by_hash.return_value = None
        mock_store_cls.return_value = mock_store
        result = runner.invoke(
            app,
            [
                "diff",
                "--db",
                "sqlite:///x.db",
                "--snapshot",
                "abcdef12",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 2
        out = _strip_ansi(result.output)
        assert "snapshot not found" in out.lower()
        assert "abcdef12" in out
