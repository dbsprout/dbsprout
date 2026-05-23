"""DBSprout TUI screen widgets (optional ``[tui]`` extra).

Each screen is a self-contained Textual widget that reads from the shared,
read-only "visual surface" sources (``.dbsprout/state.db`` for progress, the
latest schema snapshot for the schema browser) — never importing CLI or
generation code.

Intentionally does *not* import the screen modules at package import time: each
imports Textual eagerly, and pulling them in here would defeat the lazy-import
contract that keeps ``dbsprout`` CLI startup under 500 ms. Import the concrete
widget (e.g. ``dbsprout.tui.screens.schema.SchemaBrowser``) directly when needed.
"""
