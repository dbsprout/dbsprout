"""Tests for the dbsprout doctor CLI command."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from dbsprout.cli.app import app

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def test_doctor_help() -> None:
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "doctor" in _plain(result.output).lower()


def test_doctor_runs_and_reports() -> None:
    result = runner.invoke(app, ["doctor"])
    out = _plain(result.output)
    assert result.exit_code in {0, 1}
    assert "Environment" in out
    assert "Summary" in out or "passed" in out.lower()


def test_doctor_bad_db_exits_nonzero() -> None:
    result = runner.invoke(app, ["doctor", "--db", "notadialect://x"])
    assert result.exit_code == 1
