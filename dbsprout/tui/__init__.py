"""DBSprout Textual terminal UI package (optional ``[tui]`` extra).

Intentionally does *not* import :mod:`dbsprout.tui.app` at package import time:
that module imports Textual eagerly, and pulling it in here would defeat the
lazy-import contract that keeps ``dbsprout`` CLI startup under 500 ms. Import
``dbsprout.tui.app.DBSproutApp`` directly when you need the app.
"""
