"""DBSprout Textual TUI application skeleton (S-086).

A tabbed terminal UI with Progress, Schema, Quality, and Settings tabs. The
Progress tab hosts the live generation-progress screen (S-087), the Schema tab
hosts the live schema browser (S-088), and the Quality tab hosts the live
quality-results table (S-089); the Settings tab body is a placeholder.

Textual is a heavy optional dependency (the ``[tui]`` extra). It is imported at
module top-level *here* because this module is itself only imported lazily by
the ``dbsprout tui`` CLI command — never on the hot CLI startup path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.widgets import Footer, Header, Static, TabbedContent, TabPane

from dbsprout.tui.screens.progress import ProgressScreen
from dbsprout.tui.screens.quality import QualityScreen
from dbsprout.tui.screens.schema import SchemaBrowser

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema

# (tab title, pane id, placeholder body) for the not-yet-built tabs. The
# Progress (S-087), Schema (S-088) and Quality (S-089) tabs host live widgets
# and are composed separately. The per-tab switch keys live in ``BINDINGS``
# below (Textual renders them into the help footer).
_PLACEHOLDER_TABS: tuple[tuple[str, str, str], ...] = (
    ("Settings", "settings", "Settings — coming soon."),
)


def _load_latest_schema() -> DatabaseSchema | None:
    """Best-effort load of the most recent schema snapshot for the Schema tab."""
    from dbsprout.migrate.snapshot import SnapshotStore  # noqa: PLC0415

    return SnapshotStore().load_latest()


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

    def __init__(self, schema: DatabaseSchema | None = None) -> None:
        """Build the app, optionally injecting a schema for the Schema tab.

        When *schema* is ``None`` the schema is loaded from the latest snapshot
        on mount; tests inject an in-memory schema to avoid touching disk.
        """
        super().__init__()
        self._schema = schema

    def compose(self) -> ComposeResult:
        """Build the header, tabbed body, and help footer.

        Tab order (Progress, Schema, Quality, Settings) is preserved for the
        keybindings and help footer.
        """
        if self._schema is None:
            self._schema = _load_latest_schema()
        yield Header()
        with TabbedContent():
            with TabPane("Progress", id="progress"):
                yield ProgressScreen()
            with TabPane("Schema", id="schema"):
                yield SchemaBrowser(self._schema)
            with TabPane("Quality", id="quality"):
                yield QualityScreen()
            for title, pane_id, body in _PLACEHOLDER_TABS:
                with TabPane(title, id=pane_id):
                    yield Static(body)
        yield Footer()

    def action_show_tab(self, pane_id: str) -> None:
        """Switch the active tab (bound to the per-tab keybindings)."""
        self.query_one(TabbedContent).active = pane_id
