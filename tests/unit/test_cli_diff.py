"""Tests for dbsprout diff CLI command."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from dbsprout.cli.app import app

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
