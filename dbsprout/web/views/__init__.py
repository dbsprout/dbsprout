"""Web dashboard view modules (S-093+).

Each module here exposes its own :class:`~fastapi.APIRouter`, registered in
``dbsprout.web.app.create_app`` inside a story-delimited region block. View
modules read telemetry from the SQLite state layer via ``app.state.get_state_db``
and render Jinja2 templates from ``dbsprout/web/templates`` — they never import
generation/CLI code.
"""

from __future__ import annotations
