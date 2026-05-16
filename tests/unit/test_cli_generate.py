"""Tests for dbsprout generate CLI command."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Repo root = three parents up from this file (tests/unit/<file>).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Artifacts the `generate` CLI command leaks into the *current working
# directory* when a test invokes it without chdir'ing into a tmp dir
# (S-100 / DBS-120). The CWD-relative snapshot/config behaviour is
# intentional for real users; tests must isolate CWD instead. This
# module-scoped autouse guard is a standing tripwire so a future
# CWD-naive test in THIS module cannot silently re-introduce the leak
# (the bug was lost twice across /parallel-stories before being fixed).
_LEAK_ARTIFACTS = (".dbsprout", "test.db")


@pytest.fixture(autouse=True)
def _no_repo_root_pollution() -> object:
    """Fail any test in this module that leaks generate artifacts to repo root.

    Only artifacts that did not exist before the test are flagged, so a
    developer's pre-existing local ``.dbsprout/`` is not punished — only
    newly-created pollution from a CWD-naive invocation.
    """
    preexisting = {name for name in _LEAK_ARTIFACTS if (_REPO_ROOT / name).exists()}
    yield
    leaked = sorted(
        name for name in _LEAK_ARTIFACTS if name not in preexisting and (_REPO_ROOT / name).exists()
    )
    if leaked:
        pytest.fail(
            f"Test leaked generate artifacts into the repo root ({_REPO_ROOT}): "
            f"{leaked}. The `generate` CLI was invoked without "
            "`monkeypatch.chdir(tmp_path)` (see S-100 / DBS-120).",
            pytrace=False,
        )


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


class TestGenerateIncrementalFlags:
    def test_incremental_flag_in_help(self) -> None:
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--incremental" in _strip_ansi(result.output)

    def test_snapshot_flag_in_help(self) -> None:
        result = runner.invoke(app, ["generate", "--help"])
        assert "--snapshot" in _strip_ansi(result.output)

    def test_file_flag_in_help(self) -> None:
        result = runner.invoke(app, ["generate", "--help"])
        assert "--file" in _strip_ansi(result.output)


class TestGenerateRequiresSchema:
    def test_errors_without_schema(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Must error if no schema snapshot exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            ["generate", "--output-dir", str(tmp_path / "seeds")],
        )
        assert result.exit_code != 0


class TestGenerateProducesOutput:
    def test_end_to_end_sql(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Full generate with SQL output."""
        monkeypatch.chdir(tmp_path)
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

    def test_end_to_end_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Full generate with CSV output."""
        monkeypatch.chdir(tmp_path)
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

    def test_end_to_end_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Full generate with JSON output."""
        monkeypatch.chdir(tmp_path)
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

    def test_end_to_end_jsonl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Full generate with JSONL output."""
        monkeypatch.chdir(tmp_path)
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

    @pytest.mark.skipif(
        importlib.util.find_spec("polars") is None,
        reason="polars not installed (optional [data] extra)",
    )
    def test_parquet_format(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--output-format parquet should produce .parquet files."""
        monkeypatch.chdir(tmp_path)
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

    def test_invalid_format_errors(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid output format should exit with error."""
        monkeypatch.chdir(tmp_path)
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
    def test_direct_format_requires_db(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--output-format direct without --db must error."""
        monkeypatch.chdir(tmp_path)
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

    def test_direct_sqlite_uses_sa_batch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--output-format direct with sqlite:// uses SaBatchWriter."""
        from unittest.mock import MagicMock, patch  # noqa: PLC0415

        monkeypatch.chdir(tmp_path)
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


class TestGenerateRecordsState:
    """S-080: every ``generate`` records a run to the state DB."""

    def test_state_db_written(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from dbsprout.state.db import StateDB  # noqa: PLC0415

        monkeypatch.chdir(tmp_path)
        project_dir = _write_schema(tmp_path)

        result = runner.invoke(
            app,
            [
                "generate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--output-dir",
                str(project_dir / "seeds"),
                "--rows",
                "3",
                "--seed",
                "7",
            ],
        )

        assert result.exit_code == 0
        state_path = tmp_path / ".dbsprout" / "state.db"
        assert state_path.exists()
        runs = StateDB(state_path).get_runs()
        assert len(runs) == 1
        assert runs[0].engine == "heuristic"
        assert runs[0].seed == 7
        assert runs[0].total_rows == 3
        assert {s.table_name for s in runs[0].table_stats} == {"items"}
        assert len(runs[0].quality_results) >= 1

    def test_state_failure_does_not_break_generate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dbsprout.state import writer  # noqa: PLC0415

        monkeypatch.chdir(tmp_path)
        project_dir = _write_schema(tmp_path)

        def _boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("state disk full")

        monkeypatch.setattr(writer, "StateDB", _boom)

        result = runner.invoke(
            app,
            [
                "generate",
                "--schema-snapshot",
                str(project_dir / ".dbsprout" / "schema.json"),
                "--output-dir",
                str(project_dir / "seeds"),
                "--rows",
                "3",
            ],
        )

        assert result.exit_code == 0
        assert list((project_dir / "seeds").glob("*.sql"))
