"""Unit tests for the CLI error handler (S-076)."""

from __future__ import annotations

import sys

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from dbsprout.cli.app import app, run
from dbsprout.cli.error_handler import format_error_panel, handle_cli_errors
from dbsprout.errors import (
    ConnectionError,  # noqa: A004 — intentional user-facing name
    DBSproutError,
)

runner = CliRunner()


def _render(panel: object) -> str:
    console = Console(width=100, record=True)
    console.print(panel)
    return console.export_text()


def test_panel_contains_what_why_fix() -> None:
    err = DBSproutError(what="W happened", why="Y reason", fix="F remedy")
    text = _render(format_error_panel(err))
    assert "W happened" in text
    assert "Y reason" in text
    assert "F remedy" in text


def test_panel_scrubs_db_password() -> None:
    url = "postgresql://user:s3cret@host/db"
    err = ConnectionError(
        what="Could not connect.",
        why=f"Failed for {url}",
        fix="Check the URL.",
    )
    text = _render(format_error_panel(err, source_url=url))
    assert "s3cret" not in text


def test_handle_cli_errors_raises_typer_exit_with_code() -> None:
    err = DBSproutError(what="w", why="y", fix="f", exit_code=2)
    with pytest.raises(typer.Exit) as exc_info, handle_cli_errors(verbose=False):
        raise err
    assert exc_info.value.exit_code == 2


def test_handle_cli_errors_passes_through_clean_block() -> None:
    with handle_cli_errors(verbose=False):
        value = 1 + 1
    assert value == 2


def test_handle_cli_errors_wraps_unexpected_error_non_verbose() -> None:
    with pytest.raises(typer.Exit) as exc_info, handle_cli_errors(verbose=False):
        raise RuntimeError("boom")
    assert exc_info.value.exit_code == 1


def test_generic_handler_does_not_leak_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-verbose generic path must not echo raw exception strings.

    Library exceptions (e.g. SQLAlchemy) can embed connection URLs with
    passwords; only the exception class name is safe to surface.
    """
    eh = sys.modules["dbsprout.cli.error_handler"]

    rec = Console(width=100, record=True)
    monkeypatch.setattr(eh, "console", rec)
    leaky_message = "could not connect to postgresql://user:s3cr3t@host/db"
    with pytest.raises(typer.Exit), handle_cli_errors(verbose=False):
        raise RuntimeError(leaky_message)
    text = rec.export_text()
    assert "s3cr3t" not in text
    assert "RuntimeError" in text


def test_handle_cli_errors_reraises_unexpected_error_verbose() -> None:
    with pytest.raises(RuntimeError, match="boom"), handle_cli_errors(verbose=True):
        raise RuntimeError("boom")


def test_handle_cli_errors_verbose_prints_traceback_for_dbsprout_error() -> None:
    err = DBSproutError(what="w", why="y", fix="f")
    with pytest.raises(typer.Exit), handle_cli_errors(verbose=True):
        raise err


def test_handle_cli_errors_passes_through_typer_exit() -> None:
    with pytest.raises(typer.Exit) as exc_info, handle_cli_errors(verbose=False):
        raise typer.Exit(code=3)
    assert exc_info.value.exit_code == 3


def test_verbose_flag_is_accepted_by_app() -> None:
    result = runner.invoke(app, ["--verbose", "--help"])
    assert result.exit_code == 0


def test_verbose_flag_short_form_is_accepted() -> None:
    result = runner.invoke(app, ["-v", "--help"])
    assert result.exit_code == 0


def test_run_entrypoint_is_callable() -> None:
    assert callable(run)


def test_run_invokes_app_non_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    app_module = sys.modules["dbsprout.cli.app"]

    called: dict[str, bool] = {}

    def fake_app() -> None:
        called["ran"] = True

    monkeypatch.setattr(app_module, "app", fake_app)
    monkeypatch.setattr("sys.argv", ["dbsprout", "init"])
    run()
    assert called["ran"] is True


def test_run_renders_panel_for_dbsprout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_module = sys.modules["dbsprout.cli.app"]

    def fake_app() -> None:
        raise DBSproutError(what="boom", why="because", fix="do this", exit_code=7)

    monkeypatch.setattr(app_module, "app", fake_app)
    monkeypatch.setattr("sys.argv", ["dbsprout", "--verbose", "init"])
    with pytest.raises(typer.Exit) as exc_info:
        run()
    assert exc_info.value.exit_code == 7
