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
        assert "passed" in parsed
        assert "integrity" in parsed
        assert "checks" in parsed["integrity"]
        assert isinstance(parsed["integrity"]["checks"], list)


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


class TestValidateFidelity:
    def test_reference_data_rich(self, tmp_path: Path) -> None:
        """--reference-data with Rich output should show fidelity metrics."""
        project_dir = _write_schema(tmp_path)

        # Write reference CSV
        ref_dir = tmp_path / "reference"
        ref_dir.mkdir()
        csv_path = ref_dir / "items.csv"
        csv_path.write_text("id,name\n1,Widget\n2,Gadget\n3,Doohickey\n")

        result = runner.invoke(
            app,
            [
                "validate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--rows",
                "3",
                "--reference-data",
                str(ref_dir),
            ],
        )

        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "Fidelity" in output or "fidelity" in output.lower()

    def test_reference_data_json(self, tmp_path: Path) -> None:
        """--reference-data with JSON output should include fidelity section."""
        project_dir = _write_schema(tmp_path)

        ref_dir = tmp_path / "reference"
        ref_dir.mkdir()
        csv_path = ref_dir / "items.csv"
        csv_path.write_text("id,name\n1,Alpha\n2,Beta\n3,Gamma\n")

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
                "--reference-data",
                str(ref_dir),
            ],
        )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "fidelity" in parsed
        assert "overall_score" in parsed["fidelity"]
        assert "metrics" in parsed["fidelity"]

    def test_reference_data_single_file(self, tmp_path: Path) -> None:
        """--reference-data with a single CSV file."""
        project_dir = _write_schema(tmp_path)

        csv_path = tmp_path / "items.csv"
        csv_path.write_text("id,name\n1,Widget\n2,Gadget\n3,Doohickey\n")

        result = runner.invoke(
            app,
            [
                "validate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--rows",
                "3",
                "--reference-data",
                str(csv_path),
            ],
        )

        assert result.exit_code == 0

    def test_reference_data_help(self) -> None:
        """--reference-data should be a recognised validate option."""
        result = runner.invoke(app, ["validate", "--reference-data", "dummy.csv"])
        # Exit code 2 = unknown option; anything else means the flag was accepted.
        assert result.exit_code != 2

    def test_missing_reference_file(self, tmp_path: Path) -> None:
        """Missing reference data path should not crash."""
        project_dir = _write_schema(tmp_path)

        result = runner.invoke(
            app,
            [
                "validate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--rows",
                "3",
                "--reference-data",
                str(tmp_path / "nonexistent"),
            ],
        )

        assert result.exit_code == 0
