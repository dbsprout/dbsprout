"""JSON ``/api/*`` routers for the web server (S-112+).

This package holds every JSON endpoint the React Workbench SPA consumes — both
the read-only telemetry reads (``insights_api``: runs/quality/costs) and the
write APIs that mutate the in-memory
:class:`~dbsprout.web.workspace.Workspace` session (connect, schema load/paste,
spec edits, generate, regenerate, insert, export, validate, …). Each router owns
its own :class:`~fastapi.APIRouter` and is registered by
:func:`dbsprout.web.app.create_app` inside a region-delimited block so sibling
stories can extend ``create_app`` with clean union merges. (The legacy
server-rendered ``dbsprout.web.views`` package was removed in the P1c-5 cutover.)
"""
