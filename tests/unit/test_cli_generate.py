"""Tests for dbsprout generate CLI command."""

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
    """Write a minimal schema snapshot and return config dir."""
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


class TestGenerateHelp:
    def test_help_output(self) -> None:
        """--help must show generate command options."""
        result = runner.invoke(app, ["generate", "--help"])
        output = _strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--rows" in output
        assert "--seed" in output
        assert "--output-format" in output

    def test_privacy_flag_in_help(self) -> None:
        """--privacy flag must appear in help output."""
        result = runner.invoke(app, ["generate", "--help"])
        output = _strip_ansi(result.output)
        assert "--privacy" in output


class TestGenerateRequiresSchema:
    def test_errors_without_schema(self, tmp_path: Path) -> None:
        """Must error if no schema snapshot exists."""
        result = runner.invoke(
            app,
            ["generate", "--output-dir", str(tmp_path / "seeds")],
        )
        assert result.exit_code != 0


class TestGenerateProducesOutput:
    def test_end_to_end_sql(self, tmp_path: Path) -> None:
        """Full generate with SQL output."""
        project_dir = _write_schema(tmp_path)
        seeds_dir = project_dir / "seeds"

        result = runner.invoke(
            app,
            [
                "generate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--output-dir",
                str(seeds_dir),
                "--output-format",
                "sql",
                "--rows",
                "3",
                "--seed",
                "42",
            ],
        )

        assert result.exit_code == 0
        sql_files = list(seeds_dir.glob("*.sql"))
        assert len(sql_files) == 1
        assert sql_files[0].name == "001_items.sql"

    def test_end_to_end_csv(self, tmp_path: Path) -> None:
        """Full generate with CSV output."""
        project_dir = _write_schema(tmp_path)
        seeds_dir = project_dir / "seeds"

        result = runner.invoke(
            app,
            [
                "generate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--output-dir",
                str(seeds_dir),
                "--output-format",
                "csv",
                "--rows",
                "3",
            ],
        )

        assert result.exit_code == 0
        csv_files = list(seeds_dir.glob("*.csv"))
        assert len(csv_files) == 1

    def test_end_to_end_json(self, tmp_path: Path) -> None:
        """Full generate with JSON output."""
        project_dir = _write_schema(tmp_path)
        seeds_dir = project_dir / "seeds"

        result = runner.invoke(
            app,
            [
                "generate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--output-dir",
                str(seeds_dir),
                "--output-format",
                "json",
                "--rows",
                "3",
            ],
        )

        assert result.exit_code == 0
        json_files = list(seeds_dir.glob("*.json"))
        assert len(json_files) == 1
        parsed = json.loads(json_files[0].read_text())
        assert len(parsed) == 3

    def test_end_to_end_jsonl(self, tmp_path: Path) -> None:
        """Full generate with JSONL output."""
        project_dir = _write_schema(tmp_path)
        seeds_dir = project_dir / "seeds"

        result = runner.invoke(
            app,
            [
                "generate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--output-dir",
                str(seeds_dir),
                "--output-format",
                "jsonl",
                "--rows",
                "3",
            ],
        )

        assert result.exit_code == 0
        jsonl_files = list(seeds_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        lines = jsonl_files[0].read_text().strip().split("\n")
        assert len(lines) == 3

    def test_parquet_format(self, tmp_path: Path) -> None:
        """--output-format parquet should produce .parquet files."""
        project_dir = _write_schema(tmp_path)
        seeds_dir = tmp_path / "seeds"

        result = runner.invoke(
            app,
            [
                "generate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--output-dir",
                str(seeds_dir),
                "--output-format",
                "parquet",
            ],
        )

        assert result.exit_code == 0
        parquet_files = list(seeds_dir.glob("*.parquet"))
        assert len(parquet_files) == 1

    def test_invalid_format_errors(self, tmp_path: Path) -> None:
        """Invalid output format should exit with error."""
        project_dir = _write_schema(tmp_path)

        result = runner.invoke(
            app,
            [
                "generate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--output-dir",
                str(tmp_path / "seeds"),
                "--output-format",
                "xlsx",
            ],
        )

        assert result.exit_code != 0


class TestGenerateDirectFormat:
    def test_direct_format_requires_db(self, tmp_path: Path) -> None:
        """--output-format direct without --db must error."""
        project_dir = _write_schema(tmp_path)

        result = runner.invoke(
            app,
            [
                "generate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--output-format",
                "direct",
                "--rows",
                "3",
            ],
        )
        output = _strip_ansi(result.output)
        assert result.exit_code != 0
        assert "--db" in output

    def test_direct_format_in_help(self) -> None:
        """--output-format help text mentions direct."""
        result = runner.invoke(app, ["generate", "--help"])
        output = _strip_ansi(result.output)
        assert "direct" in output

    def test_db_flag_in_help(self) -> None:
        """--db flag must appear in generate help."""
        result = runner.invoke(app, ["generate", "--help"])
        output = _strip_ansi(result.output)
        assert "--db" in output

    def test_direct_sqlite_uses_sa_batch(self, tmp_path: Path) -> None:
        """--output-format direct with sqlite:// uses SaBatchWriter."""
        from unittest.mock import MagicMock, patch  # noqa: PLC0415

        project_dir = _write_schema(tmp_path)

        mock_writer = MagicMock()
        mock_writer.write.return_value = MagicMock(
            total_rows=3, tables_inserted=1, duration_seconds=0.01
        )

        with patch(
            "dbsprout.output.sa_batch.SaBatchWriter",
            return_value=mock_writer,
        ):
            result = runner.invoke(
                app,
                [
                    "generate",
                    "--schema-snapshot",
                    str(project_dir / ".dbsprout" / "schema.json"),
                    "--output-format",
                    "direct",
                    "--db",
                    "sqlite:///test.db",
                    "--rows",
                    "3",
                ],
            )

        mock_writer.write.assert_called_once()
        assert result.exit_code == 0
