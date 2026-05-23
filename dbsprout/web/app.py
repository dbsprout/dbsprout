"""FastAPI web dashboard application factory (S-090).

The dashboard is a *read-only* surface over the SQLite state layer
(``.dbsprout/state.db``, S-079). It never imports CLI/generation code and the
CLI never imports this module at startup — ``dbsprout serve`` lazy-imports it
(see :mod:`dbsprout.cli.serve`). FastAPI/uvicorn ship in the optional ``[web]``
extra.

:func:`create_app` is a factory (not a module-global singleton) so tests can
build isolated apps pointed at a temporary state DB. A module-level
``app = create_app()`` is exported too, so ``uvicorn dbsprout.web.app:app``
works for production serving.

The factory wires three things siblings rely on:

* ``app.state.templates`` — the shared :class:`~fastapi.templating.Jinja2Templates`
  environment (templates live in ``dbsprout/web/templates``).
* ``app.state.get_state_db`` — a zero-arg factory returning a fresh
  :class:`~dbsprout.state.db.StateDB` per request (cheap; opens a WAL connection).
* the shared :data:`~dbsprout.web.routes.router`, included via
  ``app.include_router`` — siblings append their handlers there.

Static assets (``dbsprout/web/static``) are mounted at ``/static``.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dbsprout.state.db import StateDB
from dbsprout.web.routes import router

_PACKAGE_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _PACKAGE_DIR / "templates"
_STATIC_DIR = _PACKAGE_DIR / "static"

#: Environment variable overriding the state-DB location (used by tests and by
#: anyone running the dashboard from outside the project root).
STATE_DB_ENV = "DBSPROUT_STATE_DB"

#: Default state-DB path, relative to the working directory.
DEFAULT_STATE_DB = Path(".dbsprout/state.db")


def _resolve_state_db_path(state_db_path: Path | str | None) -> Path:
    """Pick the state-DB path: explicit arg > ``DBSPROUT_STATE_DB`` env > default."""
    if state_db_path is not None:
        return Path(state_db_path)
    env_value = os.environ.get(STATE_DB_ENV)
    if env_value:
        return Path(env_value)
    return DEFAULT_STATE_DB


def create_app(state_db_path: Path | str | None = None) -> FastAPI:
    """Build a configured FastAPI dashboard app.

    *state_db_path* overrides where run telemetry is read from; when ``None``
    the ``DBSPROUT_STATE_DB`` env var (then :data:`DEFAULT_STATE_DB`) is used.
    """
    resolved = _resolve_state_db_path(state_db_path)

    app = FastAPI(
        title="DBSprout Dashboard",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    app.state.get_state_db = lambda: StateDB(resolved)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    app.include_router(router)
    return app


app = create_app()
