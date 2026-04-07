"""Tests for direct insertion dispatch and fallback logic."""

from __future__ import annotations

import inspect
from io import StringIO
from unittest.mock import MagicMock, patch

import click.exceptions
import pytest
from rich.console import Console
from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.cli.commands.generate import (
    _detect_direct_dialect,
    _run_direct_insert,
    generate_command,
)
from dbsprout.generate.orchestrator import GenerateResult
from dbsprout.schema.models import ColumnSchema, ColumnType, DatabaseSchema, TableSchema

# ── _detect_direct_dialect ──────────────────────────────────────────


class TestDetectDirectDialect:
    def test_postgresql_standard(self) -> None:
        assert _detect_direct_dialect("postgresql://host/db") == "postgresql"

    def test_postgres_short(self) -> None:
        assert _detect_direct_dialect("postgres://host/db") == "postgresql"

    def test_postgresql_psycopg(self) -> None:
        assert _detect_direct_dialect("postgresql+psycopg://host/db") == "postgresql"

    def test_mysql_standard(self) -> None:
        assert _detect_direct_dialect("mysql://host/db") == "mysql"

    def test_mysql_pymysql(self) -> None:
        assert _detect_direct_dialect("mysql+pymysql://host/db") == "mysql"

    def test_sqlite_file(self) -> None:
        assert _detect_direct_dialect("sqlite:///test.db") == "sqlite"

    def test_sqlite_memory(self) -> None:
        assert _detect_direct_dialect("sqlite:///:memory:") == "sqlite"

    def test_mssql(self) -> None:
        assert _detect_direct_dialect("mssql+pyodbc://host/db") == "mssql"

    def test_unknown(self) -> None:
        assert _detect_direct_dialect("oracle://host/db") == "oracle"


# ── _run_direct_insert dispatch ─────────────────────────────────────


def _make_generate_result() -> GenerateResult:
    """Create a minimal GenerateResult for dispatch tests."""
    return GenerateResult(
        tables_data={"users": [{"id": 1, "name": "Alice"}]},
        insertion_order=["users"],
        total_tables=1,
        total_rows=1,
        duration_seconds=0.01,
    )


def _make_schema() -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="users",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                    ),
                    ColumnSchema(
                        name="name",
                        data_type=ColumnType.VARCHAR,
                        nullable=False,
                        max_length=100,
                    ),
                ],
                primary_key=["id"],
            )
        ],
        dialect="sqlite",
    )


class TestDirectInsertDispatch:
    def test_sqlite_uses_sa_batch(self) -> None:
        mock_writer = MagicMock()
        mock_writer.write.return_value = MagicMock(
            total_rows=1, tables_inserted=1, duration_seconds=0.01
        )

        with patch(
            "dbsprout.output.sa_batch.SaBatchWriter",
            return_value=mock_writer,
        ):
            _run_direct_insert(
                _make_generate_result(),
                _make_schema(),
                ["users"],
                "sqlite:///test.db",
            )

        mock_writer.write.assert_called_once()

    def test_unknown_dialect_uses_sa_batch(self) -> None:
        mock_writer = MagicMock()
        mock_writer.write.return_value = MagicMock(
            total_rows=1, tables_inserted=1, duration_seconds=0.01
        )

        with patch(
            "dbsprout.output.sa_batch.SaBatchWriter",
            return_value=mock_writer,
        ):
            _run_direct_insert(
                _make_generate_result(),
                _make_schema(),
                ["users"],
                "oracle://host/db",
            )

        mock_writer.write.assert_called_once()

    def test_pg_fallback_when_psycopg_missing(self) -> None:
        mock_writer = MagicMock()
        mock_writer.write.return_value = MagicMock(
            total_rows=1, tables_inserted=1, duration_seconds=0.01
        )

        with (
            patch.dict("sys.modules", {"psycopg": None}),
            patch(
                "dbsprout.output.sa_batch.SaBatchWriter",
                return_value=mock_writer,
            ),
        ):
            _run_direct_insert(
                _make_generate_result(),
                _make_schema(),
                ["users"],
                "postgresql://host/db",
            )

        mock_writer.write.assert_called_once()

    def test_mysql_fallback_when_pymysql_missing(self) -> None:
        mock_writer = MagicMock()
        mock_writer.write.return_value = MagicMock(
            total_rows=1, tables_inserted=1, duration_seconds=0.01
        )

        with (
            patch.dict("sys.modules", {"pymysql": None}),
            patch(
                "dbsprout.output.sa_batch.SaBatchWriter",
                return_value=mock_writer,
            ),
        ):
            _run_direct_insert(
                _make_generate_result(),
                _make_schema(),
                ["users"],
                "mysql://host/db",
            )

        mock_writer.write.assert_called_once()

    def test_reports_selected_method(self) -> None:
        mock_writer = MagicMock()
        mock_writer.write.return_value = MagicMock(
            total_rows=5, tables_inserted=1, duration_seconds=0.05
        )

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=False)

        with (
            patch(
                "dbsprout.output.sa_batch.SaBatchWriter",
                return_value=mock_writer,
            ),
            patch(
                "dbsprout.cli.commands.generate.console",
                test_console,
            ),
        ):
            _run_direct_insert(
                _make_generate_result(),
                _make_schema(),
                ["users"],
                "sqlite:///test.db",
            )

        output = buf.getvalue()
        assert "Insert method:" in output
        assert "SQLAlchemy batch INSERT" in output


# ── --insert-method CLI flag ───────────────────────────────────────


class TestInsertMethodCLI:
    def test_generate_help_shows_insert_method(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["generate", "--help"])
        assert "--insert-method" in result.output

    def test_generate_command_has_insert_method_param(self) -> None:
        sig = inspect.signature(generate_command)
        assert "insert_method" in sig.parameters
        assert sig.parameters["insert_method"].default == "auto"

    def test_invalid_insert_method_raises(self) -> None:
        """Calling generate_command with an invalid insert_method should exit(1)."""
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "generate",
                "--schema-snapshot",
                "/nonexistent/schema.json",
                "--insert-method",
                "bogus",
            ],
        )
        # Should fail due to invalid insert method (before even loading schema)
        assert result.exit_code != 0
        assert "Invalid --insert-method" in result.output


class TestInsertMethodValidation:
    """Test method/dialect incompatibility guards."""

    def test_copy_on_non_pg_raises(self) -> None:
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=False)
        with (
            patch("dbsprout.cli.commands.generate.console", test_console),
            pytest.raises((SystemExit, click.exceptions.Exit)),
        ):
            _run_direct_insert(
                _make_generate_result(),
                _make_schema(),
                ["users"],
                "sqlite:///test.db",
                insert_method="copy",
            )
        assert "copy" in buf.getvalue().lower()
        assert "PostgreSQL" in buf.getvalue()

    def test_load_data_on_non_mysql_raises(self) -> None:
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=False)
        with (
            patch("dbsprout.cli.commands.generate.console", test_console),
            pytest.raises((SystemExit, click.exceptions.Exit)),
        ):
            _run_direct_insert(
                _make_generate_result(),
                _make_schema(),
                ["users"],
                "postgresql://host/db",
                insert_method="load_data",
            )
        assert "load_data" in buf.getvalue().lower()
        assert "MySQL" in buf.getvalue()

    def test_batch_forces_sa_batch(self) -> None:
        mock_writer = MagicMock()
        mock_writer.write.return_value = MagicMock(
            total_rows=5,
            tables_inserted=1,
            duration_seconds=0.05,
        )
        with patch(
            "dbsprout.output.sa_batch.SaBatchWriter",
            return_value=mock_writer,
        ):
            _run_direct_insert(
                _make_generate_result(),
                _make_schema(),
                ["users"],
                "postgresql://host/db",
                insert_method="batch",
            )
        mock_writer.write.assert_called_once()
