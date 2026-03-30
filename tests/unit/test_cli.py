"""Smoke tests for the DBSprout CLI scaffold."""

from rich.console import Console
from typer.testing import CliRunner

from dbsprout import __version__
from dbsprout.cli.app import app
from dbsprout.cli.console import console

runner = CliRunner()


def test_cli_help_returns_zero() -> None:
    """Verify `dbsprout --help` exits cleanly."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_cli_help_contains_project_name() -> None:
    """Verify help text mentions dbsprout."""
    result = runner.invoke(app, ["--help"])
    assert "dbsprout" in result.output.lower()


def test_version_importable() -> None:
    """Verify the package version is accessible."""
    assert __version__ == "0.1.0"


def test_console_is_rich_console() -> None:
    """Verify the shared console singleton is a Rich Console."""
    assert isinstance(console, Console)
