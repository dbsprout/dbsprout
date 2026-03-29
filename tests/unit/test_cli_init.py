"""Tests for dbsprout init CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import sqlalchemy.exc

if TYPE_CHECKING:
    from pathlib import Path

from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.schema.graph import FKGraph, ResolvedGraph
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

runner = CliRunner()


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
        assert "Introspect" in result.output
        assert "--db" in result.output


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
        assert "users" in result.output
        assert "orders" in result.output

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
        assert "Insertion Order" in result.output


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
        assert "error" in result.output.lower() or "Error" in result.output

    def test_no_args(self) -> None:
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1
        assert "Provide --db" in result.output

    def test_file_not_implemented(self) -> None:
        result = runner.invoke(app, ["init", "--file", "schema.sql"])
        assert result.exit_code == 1
        assert "S-010" in result.output or "DDL" in result.output

    @patch("dbsprout.cli.commands.init.introspect")
    def test_empty_db_warns(self, mock_introspect: MagicMock, tmp_path: Path) -> None:
        mock_introspect.return_value = DatabaseSchema(tables=[], dialect="sqlite")

        result = runner.invoke(
            app, ["init", "--db", "sqlite:///empty.db", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "no tables" in result.output.lower() or "No tables" in result.output
        # TOML should still be written
        assert (tmp_path / "dbsprout.toml").exists()
        # No snapshot for empty schema
        snapshot_dir = tmp_path / ".dbsprout" / "snapshots"
        snapshots = list(snapshot_dir.glob("*.json")) if snapshot_dir.exists() else []
        assert len(snapshots) == 0
