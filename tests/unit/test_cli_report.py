"""Tests for the ``dbsprout report`` standalone command (S-085)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.state import RunRecord, StateDB, TableStats

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# CLI/Rich CI flake guard: no TTY in CI wraps/ANSI-fragments tokens. Every
# output assertion strips ANSI and forces a wide, colourless terminal.
_WIDE_ENV = {"COLUMNS": "200", "NO_COLOR": "1"}

_NO_RUNS_ERROR = "No generation runs found. Run `dbsprout generate --report` first."


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _record_run(db_path: Path, *, total_rows: int, table_name: str) -> int:
    """Record a single run into a state DB and return its id."""
    state = StateDB(db_path)
    now = datetime.now(tz=timezone.utc)
    return state.record_run(
        RunRecord(
            started_at=now,
            completed_at=now,
            duration_ms=10,
            engine="heuristic",
            total_rows=total_rows,
            total_tables=1,
            seed=42,
            table_stats=[TableStats(table_name=table_name, row_count=total_rows)],
        )
    )


class TestReportCommandNoState:
    def test_no_state_db_file_errors(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["report"], env=_WIDE_ENV)
        assert result.exit_code == 1
        assert _NO_RUNS_ERROR in _strip_ansi(result.output)

    def test_empty_state_db_errors(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        # State DB exists but has zero recorded runs.
        StateDB(tmp_path / ".dbsprout" / "state.db")
        result = runner.invoke(app, ["report"], env=_WIDE_ENV)
        assert result.exit_code == 1
        assert _NO_RUNS_ERROR in _strip_ansi(result.output)


class TestReportCommandNewestRun:
    def test_generates_report_from_newest_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _record_run(tmp_path / ".dbsprout" / "state.db", total_rows=111, table_name="alpha")
        result = runner.invoke(app, ["report"], env=_WIDE_ENV)
        assert result.exit_code == 0, result.output
        report = tmp_path / "seeds" / "report.html"
        assert report.exists()
        html = report.read_text(encoding="utf-8")
        assert "alpha" in html
        out = _strip_ansi(result.output)
        assert "Report saved to" in out
        assert "report.html" in out

    def test_custom_output_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        _record_run(tmp_path / ".dbsprout" / "state.db", total_rows=5, table_name="beta")
        dest = tmp_path / "out" / "custom.html"
        result = runner.invoke(app, ["report", "--output", str(dest)], env=_WIDE_ENV)
        assert result.exit_code == 0, result.output
        assert dest.exists()
        assert "custom.html" in _strip_ansi(result.output)


class TestReportCommandRunId:
    def test_specific_historic_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        db = tmp_path / ".dbsprout" / "state.db"
        first_id = _record_run(db, total_rows=10, table_name="oldtable")
        _record_run(db, total_rows=99, table_name="newtable")
        result = runner.invoke(app, ["report", "--run-id", str(first_id)], env=_WIDE_ENV)
        assert result.exit_code == 0, result.output
        html = (tmp_path / "seeds" / "report.html").read_text(encoding="utf-8")
        # The older run was selected, not the newest.
        assert "oldtable" in html
        assert "newtable" not in html

    def test_missing_run_id_errors(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        _record_run(tmp_path / ".dbsprout" / "state.db", total_rows=1, table_name="t")
        result = runner.invoke(app, ["report", "--run-id", "999"], env=_WIDE_ENV)
        assert result.exit_code == 1
        out = _strip_ansi(result.output)
        assert "999" in out
        assert "not found" in out.lower()


class TestReportCommandHelp:
    def test_report_help_lists_options(self) -> None:
        result = runner.invoke(app, ["report", "--help"], env=_WIDE_ENV)
        assert result.exit_code == 0
        out = _strip_ansi(result.output)
        assert "--output" in out
        assert "--run-id" in out

    def test_root_help_lists_report(self) -> None:
        result = runner.invoke(app, ["--help"], env=_WIDE_ENV)
        assert result.exit_code == 0
        assert "report" in _strip_ansi(result.output)
