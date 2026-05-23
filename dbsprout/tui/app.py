"""DBSprout Textual TUI application skeleton (S-086).

A tabbed terminal UI with Progress, Schema, Quality, and Settings tabs. This
story delivers the skeleton only — tab bodies are placeholders that later
stories (S-087/S-088/S-089) replace with live content sourced from
``.dbsprout/state.db``.

Textual is a heavy optional dependency (the ``[tui]`` extra). It is imported at
module top-level *here* because this module is itself only imported lazily by
the ``dbsprout tui`` CLI command — never on the hot CLI startup path.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.widgets import Footer, Header, Static, TabbedContent, TabPane

# (tab title, pane id, placeholder body). The per-tab switch keys live in
# ``BINDINGS`` below (which Textual also renders into the help footer).
_TABS: tuple[tuple[str, str, str], ...] = (
    ("Progress", "progress", "Generation progress — coming soon (S-087)."),
    ("Schema", "schema", "Schema browser — coming soon (S-088)."),
    ("Quality", "quality", "Quality metrics — coming soon (S-089)."),
    ("Settings", "settings", "Settings — coming soon."),
)


class DBSproutApp(App[None]):
    """Tabbed DBSprout terminal UI."""

    CSS_PATH = "dbsprout.tcss"

    TITLE = "DBSprout"
    SUB_TITLE = "terminal UI"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("p", "show_tab('progress')", "Progress"),
        Binding("s", "show_tab('schema')", "Schema"),
        Binding("u", "show_tab('quality')", "Quality"),
        Binding("g", "show_tab('settings')", "Settings"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """Build the header, tabbed body, and help footer."""
        yield Header()
        with TabbedContent():
            for title, pane_id, body in _TABS:
                with TabPane(title, id=pane_id):
                    yield Static(body)
        yield Footer()

    def action_show_tab(self, pane_id: str) -> None:
        """Switch the active tab (bound to the per-tab keybindings)."""
        self.query_one(TabbedContent).active = pane_id
