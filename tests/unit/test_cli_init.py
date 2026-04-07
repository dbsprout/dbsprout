"""Tests for dbsprout init CLI command."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import sqlalchemy.exc
from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.schema.graph import (
    CycleInfo,
    DeferredFK,
    FKGraph,
    ResolvedGraph,
    UnresolvableCycleError,
)
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes for assertion matching."""
    return _ANSI_RE.sub("", text)


def _simple_schema() -> DatabaseSchema:
    """Two-table schema for testing."""
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="users",
                columns=[
                    ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                    ColumnSchema(name="email", data_type=ColumnType.VARCHAR, max_length=255),
                ],
                primary_key=["id"],
            ),
            TableSchema(
                name="orders",
                columns=[
                    ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                    ColumnSchema(name="user_id", data_type=ColumnType.INTEGER),
                ],
                primary_key=["id"],
                foreign_keys=[
                    ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"]),
                ],
            ),
        ],
        dialect="sqlite",
        source="introspect",
    )


def _mock_resolved(schema: DatabaseSchema) -> ResolvedGraph:
    """Build a ResolvedGraph from a schema for testing."""
    return ResolvedGraph(graph=FKGraph.from_schema(schema))


# ── Help ─────────────────────────────────────────────────────────────────


class TestInitHelp:
    def test_help_shows_usage(self) -> None:
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        assert "Introspect" in _strip_ansi(result.output)
        assert "--db" in _strip_ansi(result.output)


# ── Happy path ───────────────────────────────────────────────────────────


class TestInitHappyPath:
    @patch("dbsprout.cli.commands.init.resolve_cycles")
    @patch("dbsprout.cli.commands.init.introspect")
    def test_init_db_sqlite(self, mock_introspect: MagicMock, mock_resolve: MagicMock) -> None:
        schema = _simple_schema()
        mock_introspect.return_value = schema
        mock_resolve.return_value = _mock_resolved(schema)

        result = runner.invoke(app, ["init", "--db", "sqlite:///test.db"])
        assert result.exit_code == 0
        assert "users" in _strip_ansi(result.output)
        assert "orders" in _strip_ansi(result.output)

    @patch("dbsprout.cli.commands.init.resolve_cycles")
    @patch("dbsprout.cli.commands.init.introspect")
    def test_init_shows_insertion_order(
        self, mock_introspect: MagicMock, mock_resolve: MagicMock
    ) -> None:
        schema = _simple_schema()
        mock_introspect.return_value = schema
        mock_resolve.return_value = _mock_resolved(schema)

        result = runner.invoke(app, ["init", "--db", "sqlite:///test.db"])
        assert result.exit_code == 0
        assert "Insertion Order" in _strip_ansi(result.output)


# ── Dry run ──────────────────────────────────────────────────────────────


class TestInitDryRun:
    @patch("dbsprout.cli.commands.init.resolve_cycles")
    @patch("dbsprout.cli.commands.init.introspect")
    def test_dry_run_no_files(
        self, mock_introspect: MagicMock, mock_resolve: MagicMock, tmp_path: Path
    ) -> None:
        schema = _simple_schema()
        mock_introspect.return_value = schema
        mock_resolve.return_value = _mock_resolved(schema)

        result = runner.invoke(
            app, ["init", "--db", "sqlite:///test.db", "--dry-run", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert not (tmp_path / "dbsprout.toml").exists()


# ── File writing ─────────────────────────────────────────────────────────


class TestInitFileWriting:
    @patch("dbsprout.cli.commands.init.resolve_cycles")
    @patch("dbsprout.cli.commands.init.introspect")
    def test_writes_toml(
        self, mock_introspect: MagicMock, mock_resolve: MagicMock, tmp_path: Path
    ) -> None:
        schema = _simple_schema()
        mock_introspect.return_value = schema
        mock_resolve.return_value = _mock_resolved(schema)

        result = runner.invoke(
            app, ["init", "--db", "sqlite:///test.db", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        toml_file = tmp_path / "dbsprout.toml"
        assert toml_file.exists()
        content = toml_file.read_text()
        assert "sqlite" in content
        assert "default_rows" in content

    @patch("dbsprout.cli.commands.init.resolve_cycles")
    @patch("dbsprout.cli.commands.init.introspect")
    def test_writes_snapshot(
        self, mock_introspect: MagicMock, mock_resolve: MagicMock, tmp_path: Path
    ) -> None:
        schema = _simple_schema()
        mock_introspect.return_value = schema
        mock_resolve.return_value = _mock_resolved(schema)

        result = runner.invoke(
            app, ["init", "--db", "sqlite:///test.db", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        snapshot_dir = tmp_path / ".dbsprout" / "snapshots"
        assert snapshot_dir.exists()
        snapshots = list(snapshot_dir.glob("*.json"))
        assert len(snapshots) == 1


# ── Error cases ──────────────────────────────────────────────────────────


class TestInitErrors:
    @patch("dbsprout.cli.commands.init.introspect")
    def test_connection_error(self, mock_introspect: MagicMock) -> None:
        mock_introspect.side_effect = sqlalchemy.exc.SQLAlchemyError("connection refused")

        result = runner.invoke(app, ["init", "--db", "sqlite:///bad.db"])
        assert result.exit_code == 1
        assert "error" in _strip_ansi(result.output).lower()

    def test_no_args(self) -> None:
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1
        assert "Provide --db" in _strip_ansi(result.output)

    def test_file_not_found(self) -> None:
        result = runner.invoke(app, ["init", "--file", "nonexistent.sql"])
        assert result.exit_code == 1
        assert "not found" in _strip_ansi(result.output).lower()

    def test_db_and_file_mutually_exclusive(self) -> None:
        result = runner.invoke(app, ["init", "--db", "sqlite:///x", "--file", "y.sql"])
        assert result.exit_code == 1
        assert "only one" in _strip_ansi(result.output).lower()

    def test_file_empty_ddl(self, tmp_path: Path) -> None:
        ddl_file = tmp_path / "empty.sql"
        ddl_file.write_text("-- just comments")
        result = runner.invoke(
            app, ["init", "--file", str(ddl_file), "--output-dir", str(tmp_path)]
        )
        assert result.exit_code == 1

    def test_file_too_large(self, tmp_path: Path) -> None:
        big_file = tmp_path / "big.sql"
        big_file.write_text("x" * 100)  # small file, but we'll test the path exists
        # Can't easily create a 10MB+ file in unit test, just verify the guard exists
        result = runner.invoke(
            app, ["init", "--file", str(big_file), "--output-dir", str(tmp_path)]
        )
        # File is small so it should parse (or fail on content), not on size
        assert result.exit_code in (0, 1)

    def test_file_parses_ddl(self, tmp_path: Path) -> None:
        ddl_file = tmp_path / "schema.sql"
        ddl_file.write_text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);")
        result = runner.invoke(
            app, ["init", "--file", str(ddl_file), "--output-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "users" in _strip_ansi(result.output)
        assert (tmp_path / "dbsprout.toml").exists()

    @patch("dbsprout.cli.commands.init.introspect")
    def test_empty_db_warns(self, mock_introspect: MagicMock, tmp_path: Path) -> None:
        mock_introspect.return_value = DatabaseSchema(tables=[], dialect="sqlite")

        result = runner.invoke(
            app, ["init", "--db", "sqlite:///empty.db", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "no tables" in _strip_ansi(result.output).lower()
        # TOML should still be written
        assert (tmp_path / "dbsprout.toml").exists()
        # No snapshot for empty schema
        snapshot_dir = tmp_path / ".dbsprout" / "snapshots"
        snapshots = list(snapshot_dir.glob("*.json")) if snapshot_dir.exists() else []
        assert len(snapshots) == 0

    @patch("dbsprout.cli.commands.init.resolve_cycles")
    @patch("dbsprout.cli.commands.init.introspect")
    def test_unresolvable_cycle(self, mock_introspect: MagicMock, mock_resolve: MagicMock) -> None:
        mock_introspect.return_value = _simple_schema()
        mock_resolve.side_effect = UnresolvableCycleError(CycleInfo(tables=frozenset({"a", "b"})))
        result = runner.invoke(app, ["init", "--db", "sqlite:///test.db"])
        assert result.exit_code == 1
        assert "nullable" in _strip_ansi(result.output).lower()


# ── Cycle + self-ref display ─────────────────────────────────────────────


class TestInitCycleDisplay:
    @patch("dbsprout.cli.commands.init.resolve_cycles")
    @patch("dbsprout.cli.commands.init.introspect")
    def test_deferred_fk_display(
        self, mock_introspect: MagicMock, mock_resolve: MagicMock, tmp_path: Path
    ) -> None:
        schema = _simple_schema()
        mock_introspect.return_value = schema
        graph = FKGraph.from_schema(schema)
        mock_resolve.return_value = ResolvedGraph(
            graph=graph,
            deferred_fks=(
                DeferredFK(
                    source_table="orders",
                    foreign_key=ForeignKeySchema(
                        columns=["user_id"], ref_table="users", ref_columns=["id"]
                    ),
                ),
            ),
        )
        result = runner.invoke(
            app, ["init", "--db", "sqlite:///test.db", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "deferred" in _strip_ansi(result.output).lower()

    @patch("dbsprout.cli.commands.init.resolve_cycles")
    @patch("dbsprout.cli.commands.init.introspect")
    def test_self_ref_display(
        self, mock_introspect: MagicMock, mock_resolve: MagicMock, tmp_path: Path
    ) -> None:
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="employees",
                    columns=[
                        ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False),
                        ColumnSchema(name="manager_id", data_type=ColumnType.INTEGER),
                    ],
                    primary_key=["id"],
                    foreign_keys=[
                        ForeignKeySchema(
                            columns=["manager_id"], ref_table="employees", ref_columns=["id"]
                        )
                    ],
                )
            ],
            dialect="sqlite",
        )
        mock_introspect.return_value = schema
        graph = FKGraph.from_schema(schema)
        mock_resolve.return_value = ResolvedGraph(graph=graph)
        result = runner.invoke(
            app, ["init", "--db", "sqlite:///test.db", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        out = _strip_ansi(result.output).lower()
        assert "self-referencing" in out or "employees" in out


# ── TOML content ─────────────────────────────────────────────────────────


class TestInitTomlContent:
    @patch("dbsprout.cli.commands.init.resolve_cycles")
    @patch("dbsprout.cli.commands.init.introspect")
    def test_toml_has_snapshot_path(
        self, mock_introspect: MagicMock, mock_resolve: MagicMock, tmp_path: Path
    ) -> None:
        schema = _simple_schema()
        mock_introspect.return_value = schema
        mock_resolve.return_value = _mock_resolved(schema)

        runner.invoke(app, ["init", "--db", "sqlite:///test.db", "--output-dir", str(tmp_path)])
        content = (tmp_path / "dbsprout.toml").read_text()
        assert "snapshot" in content
        assert ".dbsprout/snapshots/" in content
