"""TUI app skeleton tests (S-086).

Textual's :meth:`textual.app.App.run_test` is an async context manager that
drives the app with a headless ``Pilot``. The repo has no async pytest plugin
(and ``addopts`` is fixed), so each test wraps its pilot coroutine in
``asyncio.run(...)`` inside an ordinary synchronous test function — zero new
test dependencies.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from textual.widgets import Footer, Header, TabbedContent, TabPane

from dbsprout.tui.app import DBSproutApp

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from typing import TypeVar

    _T = TypeVar("_T")


_EXPECTED_TABS = ["Progress", "Schema", "Quality", "Settings"]
_PANE_IDS = {
    "p": "progress",
    "s": "schema",
    "u": "quality",
    "g": "settings",
}


def _run(coro: Awaitable[_T]) -> _T:
    """Run an async pilot coroutine in a fresh event loop."""
    return asyncio.run(coro)  # type: ignore[arg-type]


def test_app_composes_four_named_tabs() -> None:
    async def _scenario() -> None:
        app = DBSproutApp()
        async with app.run_test() as pilot:
            tabbed = pilot.app.query_one(TabbedContent)
            titles = [str(tabbed.get_tab(pane.id).label) for pane in tabbed.query(TabPane)]
            assert titles == _EXPECTED_TABS
            # Header + Footer present for chrome / help.
            assert pilot.app.query(Header)
            assert pilot.app.query(Footer)

    _run(_scenario())


def test_keybindings_switch_active_tab() -> None:
    async def _scenario() -> None:
        app = DBSproutApp()
        async with app.run_test() as pilot:
            tabbed = pilot.app.query_one(TabbedContent)
            for key, pane_id in _PANE_IDS.items():
                await pilot.press(key)
                assert tabbed.active == pane_id

    _run(_scenario())


def test_every_binding_has_help_description() -> None:
    # The Footer renders one entry per *shown* binding that carries a
    # description. Normalise tuple- and Binding-style entries alike.
    tuple_with_description = 3  # ("key", "action", "Description")
    shown_descriptions: list[str] = []
    for entry in DBSproutApp.BINDINGS:
        if isinstance(entry, tuple):
            description = entry[2] if len(entry) >= tuple_with_description else ""
            show = True
        else:
            description = entry.description
            show = entry.show
        if show:
            shown_descriptions.append(description)

    assert shown_descriptions, "expected at least one shown binding for the help footer"
    assert all(shown_descriptions), "every shown binding must have a non-empty description"


def test_q_quits_gracefully() -> None:
    async def _scenario() -> None:
        app = DBSproutApp()
        async with app.run_test() as pilot:
            assert pilot.app.is_running
            await pilot.press("q")
        # Exiting the context cleanly (no exception) is the graceful-exit signal.
        assert app.return_code in (0, None)

    _run(_scenario())


def test_css_file_exists_and_is_responsive() -> None:
    css_path = Path(DBSproutApp.CSS_PATH)  # type: ignore[arg-type]
    if not css_path.is_absolute():
        css_path = Path(DBSproutApp().css_path[0])  # resolved relative to app module
    assert css_path.exists(), f"missing TUI stylesheet: {css_path}"
    text = css_path.read_text(encoding="utf-8")
    assert text.strip(), "stylesheet must not be empty"
    # Fractional height makes the layout adapt to terminal size.
    assert "1fr" in text


def test_dbsprout_tui_command_runs_the_app() -> None:
    from unittest.mock import patch  # noqa: PLC0415

    from typer.testing import CliRunner  # noqa: PLC0415

    from dbsprout.cli.app import app  # noqa: PLC0415

    runner = CliRunner()
    with patch.object(DBSproutApp, "run", autospec=True) as mock_run:
        result = runner.invoke(app, ["tui"])

    assert result.exit_code == 0, result.output
    assert mock_run.call_count == 1


def test_cli_app_does_not_import_textual_eagerly() -> None:
    """Importing the CLI must not pull Textual (preserves <500ms startup)."""
    probe = "import sys\nimport dbsprout.cli.app  # noqa: F401\nprint('textual' in sys.modules)\n"
    result = subprocess.run(  # noqa: S603 - fixed argv, trusted interpreter
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == "False", (
        "importing dbsprout.cli.app eagerly imported textual; the `tui` proxy "
        "must lazy-import Textual inside the command body."
    )
