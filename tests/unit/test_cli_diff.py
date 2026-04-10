"""Tests for dbsprout diff CLI command."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes for assertion matching."""
    return _ANSI_RE.sub("", text)


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
