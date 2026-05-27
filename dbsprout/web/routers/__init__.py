"""Read-WRITE JSON API routers for the web dashboard (S-112+).

This package is the home for endpoints that *mutate* the in-memory
:class:`~dbsprout.web.workspace.Workspace` session (connect, spec edits,
generate, …) and return JSON. It is deliberately separate from
:mod:`dbsprout.web.views`, which holds the read-only Jinja2 views over the
SQLite state layer. Each router owns its own :class:`~fastapi.APIRouter` and is
registered by :func:`dbsprout.web.app.create_app` inside a region-delimited
block so sibling stories can extend ``create_app`` with clean union merges.
"""
