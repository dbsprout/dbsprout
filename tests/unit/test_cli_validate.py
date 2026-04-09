"""Tests for dbsprout validate CLI command."""

from __future__ import annotations

import importlib.util
import json
import re
from typing import TYPE_CHECKING

import pytest
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
        """--format json outputs valid JSON with QualityReport envelope."""
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
        # QualityReport envelope keys
        assert "version" in parsed
        assert "metadata" in parsed
        assert "fidelity" in parsed
        assert "detection" in parsed


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
    @pytest.mark.skipif(
        importlib.util.find_spec("scipy") is None,
        reason="scipy not installed (optional [stats] extra)",
    )
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

    @pytest.mark.skipif(
        importlib.util.find_spec("scipy") is None,
        reason="scipy not installed (optional [stats] extra)",
    )
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

    @pytest.mark.skipif(
        importlib.util.find_spec("scipy") is None,
        reason="scipy not installed (optional [stats] extra)",
    )
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


def _write_detection_schema(tmp_path: Path) -> Path:
    """Write a schema with enough numeric columns for C2ST and return project dir."""
    schema = DatabaseSchema(
        tables=[
            TableSchema(
                name="products",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                        autoincrement=True,
                    ),
                    ColumnSchema(
                        name="price",
                        data_type=ColumnType.FLOAT,
                        nullable=False,
                    ),
                    ColumnSchema(
                        name="quantity",
                        data_type=ColumnType.INTEGER,
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


def _write_detection_reference_csv(tmp_path: Path) -> Path:
    """Write reference CSV with enough rows for C2ST (>=10 per class)."""
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir(exist_ok=True)
    csv_path = ref_dir / "products.csv"
    lines = ["id,price,quantity"]
    for i in range(1, 21):
        lines.append(f"{i},{i * 10.5},{i * 2}")
    csv_path.write_text("\n".join(lines) + "\n")
    return ref_dir


class TestValidateDetection:
    def test_detection_flag_accepted(self, tmp_path: Path) -> None:
        """--detection with --reference-data should not fail with exit code 2."""
        project_dir = _write_detection_schema(tmp_path)
        ref_dir = _write_detection_reference_csv(tmp_path)

        result = runner.invoke(
            app,
            [
                "validate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--rows",
                "20",
                "--seed",
                "42",
                "--detection",
                "--reference-data",
                str(ref_dir),
            ],
        )

        assert result.exit_code != 2, f"Flag not recognised: {result.output}"

    def test_detection_without_reference(self, tmp_path: Path) -> None:
        """--detection without --reference-data should print error and exit 1."""
        project_dir = _write_detection_schema(tmp_path)

        result = runner.invoke(
            app,
            [
                "validate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--rows",
                "5",
                "--detection",
            ],
        )

        assert result.exit_code == 1
        output = _strip_ansi(result.output)
        assert "reference-data" in output.lower()

    @pytest.mark.skipif(
        importlib.util.find_spec("sklearn") is None,
        reason="sklearn not installed (optional [stats] extra)",
    )
    def test_detection_json_output(self, tmp_path: Path) -> None:
        """--detection --format json should include 'detection' key."""
        project_dir = _write_detection_schema(tmp_path)
        ref_dir = _write_detection_reference_csv(tmp_path)

        result = runner.invoke(
            app,
            [
                "validate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--rows",
                "20",
                "--seed",
                "42",
                "--detection",
                "--reference-data",
                str(ref_dir),
                "--format",
                "json",
            ],
        )

        assert result.exit_code in {0, 1}
        parsed = json.loads(result.output)
        assert "detection" in parsed
        assert "passed" in parsed["detection"]
        assert "overall_score" in parsed["detection"]
        assert "metrics" in parsed["detection"]
        assert isinstance(parsed["detection"]["metrics"], list)


class TestValidateOutputCompact:
    """Tests for --output and --compact CLI flags."""

    def test_help_shows_output_and_compact(self) -> None:
        """--help must list --output and --compact options."""
        result = runner.invoke(app, ["validate", "--help"])
        output = _strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--output" in output
        assert "--compact" in output

    def test_output_without_json_format_errors(self, tmp_path: Path) -> None:
        """--output requires --format json; without it, exit 1."""
        project_dir = _write_schema(tmp_path)
        schema_path = str(project_dir / ".dbsprout" / "schema.json")

        result = runner.invoke(
            app,
            [
                "validate",
                "--schema-snapshot",
                schema_path,
                "--rows",
                "3",
                "--output",
                str(tmp_path / "report.json"),
            ],
        )

        assert result.exit_code == 1
        output = _strip_ansi(result.output)
        assert "requires --format json" in output.lower() or "requires --format json" in output

    def test_compact_without_json_silently_ignored(self, tmp_path: Path) -> None:
        """--compact without --format json is silently ignored (exit 0)."""
        project_dir = _write_schema(tmp_path)
        schema_path = str(project_dir / ".dbsprout" / "schema.json")

        result = runner.invoke(
            app,
            [
                "validate",
                "--schema-snapshot",
                schema_path,
                "--rows",
                "3",
                "--compact",
            ],
        )

        assert result.exit_code == 0


class TestValidateJsonReport:
    """Tests for QualityReport JSON envelope, ANSI leak fix, output/compact."""

    def test_json_output_has_version_and_metadata(self, tmp_path: Path) -> None:
        """JSON output must include version and metadata with expected sub-keys."""
        project_dir = _write_schema(tmp_path)
        schema_path = str(project_dir / ".dbsprout" / "schema.json")

        result = runner.invoke(
            app,
            [
                "validate",
                "--format",
                "json",
                "--schema-snapshot",
                schema_path,
            ],
        )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "version" in parsed
        assert parsed["version"] == "1.0.0"
        assert "metadata" in parsed
        meta = parsed["metadata"]
        for key in ("timestamp", "schema_hash", "row_counts", "engine", "seed"):
            assert key in meta, f"Missing metadata key: {key}"
        assert meta["engine"] == "heuristic"
        assert meta["seed"] == 42

    def test_json_output_fidelity_null_when_absent(self, tmp_path: Path) -> None:
        """Fidelity must be null when --reference-data is not provided."""
        project_dir = _write_schema(tmp_path)
        schema_path = str(project_dir / ".dbsprout" / "schema.json")

        result = runner.invoke(
            app,
            [
                "validate",
                "--format",
                "json",
                "--schema-snapshot",
                schema_path,
            ],
        )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["fidelity"] is None

    def test_json_output_detection_null_when_absent(self, tmp_path: Path) -> None:
        """Detection must be null when --detection is not provided."""
        project_dir = _write_schema(tmp_path)
        schema_path = str(project_dir / ".dbsprout" / "schema.json")

        result = runner.invoke(
            app,
            [
                "validate",
                "--format",
                "json",
                "--schema-snapshot",
                schema_path,
            ],
        )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["detection"] is None

    def test_json_compact_no_indent(self, tmp_path: Path) -> None:
        """--compact must produce single-line JSON (no indented lines)."""
        project_dir = _write_schema(tmp_path)
        schema_path = str(project_dir / ".dbsprout" / "schema.json")

        result = runner.invoke(
            app,
            [
                "validate",
                "--format",
                "json",
                "--compact",
                "--schema-snapshot",
                schema_path,
            ],
        )

        assert result.exit_code == 0
        assert "\n  " not in result.output

    def test_json_output_to_file(self, tmp_path: Path) -> None:
        """--output writes JSON to file; stdout must not contain JSON."""
        project_dir = _write_schema(tmp_path)
        schema_path = str(project_dir / ".dbsprout" / "schema.json")
        report_path = tmp_path / "report.json"

        result = runner.invoke(
            app,
            [
                "validate",
                "--format",
                "json",
                "--output",
                str(report_path),
                "--schema-snapshot",
                schema_path,
            ],
        )

        assert result.exit_code == 0
        assert report_path.exists()
        file_content = report_path.read_text(encoding="utf-8")
        parsed = json.loads(file_content)
        assert "version" in parsed
        # JSON went to file, NOT stdout
        assert "version" not in result.output

    def test_json_output_to_unwritable_path(self, tmp_path: Path) -> None:
        """--output to a non-existent directory must exit 1."""
        project_dir = _write_schema(tmp_path)
        schema_path = str(project_dir / ".dbsprout" / "schema.json")

        result = runner.invoke(
            app,
            [
                "validate",
                "--format",
                "json",
                "--output",
                "/nonexistent-dir-xyz/report.json",
                "--schema-snapshot",
                schema_path,
            ],
        )

        assert result.exit_code == 1

    def test_json_no_ansi_escape_codes(self, tmp_path: Path) -> None:
        """JSON output must not contain ANSI escape sequences."""
        project_dir = _write_schema(tmp_path)
        schema_path = str(project_dir / ".dbsprout" / "schema.json")

        result = runner.invoke(
            app,
            [
                "validate",
                "--format",
                "json",
                "--schema-snapshot",
                schema_path,
            ],
        )

        assert result.exit_code == 0
        assert "\x1b[" not in result.output
