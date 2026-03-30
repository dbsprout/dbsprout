"""Tests for dbsprout validate CLI command."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

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

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _write_schema(tmp_path: Path) -> Path:
    """Write a minimal schema snapshot and return project dir."""
    schema = DatabaseSchema(
        tables=[
            TableSchema(
                name="items",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                        autoincrement=True,
                    ),
                    ColumnSchema(
                        name="name",
                        data_type=ColumnType.VARCHAR,
                        nullable=False,
                    ),
                ],
                primary_key=["id"],
            ),
        ],
    )
    snapshot_dir = tmp_path / ".dbsprout"
    snapshot_dir.mkdir()
    snapshot_path = snapshot_dir / "schema.json"
    snapshot_path.write_text(schema.model_dump_json(indent=2), encoding="utf-8")
    return tmp_path


class TestValidateHelp:
    def test_help_output(self) -> None:
        """--help must show validate command options."""
        result = runner.invoke(app, ["validate", "--help"])
        output = _strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--schema-snapshot" in output
        assert "--format" in output


class TestValidateRequiresSchema:
    def test_errors_without_schema(self, tmp_path: Path) -> None:
        """Must error if no schema snapshot exists."""
        result = runner.invoke(app, ["validate"])
        assert result.exit_code != 0


class TestValidateWithSchema:
    def test_validates_and_passes(self, tmp_path: Path) -> None:
        """Validates generated data, all checks pass, exit 0."""
        project_dir = _write_schema(tmp_path)

        result = runner.invoke(
            app,
            [
                "validate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--rows",
                "5",
                "--seed",
                "42",
            ],
        )

        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "PASS" in output or "pass" in output.lower()


class TestValidateJSONFormat:
    def test_json_output(self, tmp_path: Path) -> None:
        """--format json outputs valid JSON."""
        project_dir = _write_schema(tmp_path)

        result = runner.invoke(
            app,
            [
                "validate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--rows",
                "3",
                "--format",
                "json",
            ],
        )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "checks" in parsed
        assert "passed" in parsed
        assert isinstance(parsed["checks"], list)


class TestValidateSummary:
    def test_summary_shows_counts(self, tmp_path: Path) -> None:
        """Summary output shows check counts."""
        project_dir = _write_schema(tmp_path)

        result = runner.invoke(
            app,
            [
                "validate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--rows",
                "3",
            ],
        )

        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        # Should contain check count info
        assert "check" in output.lower() or "pass" in output.lower()
