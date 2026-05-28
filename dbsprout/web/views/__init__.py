"""Web dashboard view modules (S-091+).

Each module here exposes its own :class:`~fastapi.APIRouter`, registered in
``dbsprout.web.app.create_app`` inside a story-delimited region block. View
modules read telemetry from the SQLite state layer via ``app.state.get_state_db``
(and schema snapshots via ``app.state.get_snapshot_store``) and render Jinja2
templates from ``dbsprout/web/templates`` — they never import generation/CLI code.

This isolates parallel-wave stories: a new view is a new module here plus one
region-delimited ``include_router`` line in ``app.py``, never an edit to a
shared router file.
"""

from __future__ import annotations
