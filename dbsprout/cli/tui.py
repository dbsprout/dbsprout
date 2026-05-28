"""`dbsprout tui` command body — launches the Textual terminal UI (S-086).

Textual is imported lazily *inside* :func:`tui_command` so that importing the
CLI never pulls the heavy ``[tui]`` dependency (preserves the <500 ms startup
budget). The proxy in :mod:`dbsprout.cli.app` delegates here.
"""

from __future__ import annotations


def tui_command() -> None:
    """Launch the DBSprout Textual terminal UI (blocks until the user quits)."""
    from dbsprout.tui.app import DBSproutApp  # noqa: PLC0415

    DBSproutApp().run()
