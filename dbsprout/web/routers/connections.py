"""``/api/connections`` — saved-connection CRUD (P2a-2).

Persists reusable database connections to ``.dbsprout/connections.toml`` so a
user can save a target once and reload it from the Start panel. Three routes:

* ``GET  /api/connections`` — list saved connections (name + URL; **never** a
  literal password).
* ``POST /api/connections`` — save ``{ name, url }``; the literal password is
  stripped before persistence (an ``${ENV_VAR}`` reference is preserved).
* ``DELETE /api/connections/{name}`` — remove one; ``404`` typed envelope when
  the name is unknown.

All persistence + password stripping lives in the pure
:mod:`dbsprout.core.connections` helper (unit-tested in isolation); this router
is a thin boundary that validates input and selects the on-disk path. The path
is ``.dbsprout/connections.toml`` under the working directory, overridable via
``DBSPROUT_CONNECTIONS_PATH`` (used by tests and by anyone running the dashboard
from outside the project root). The module is imported only by
:mod:`dbsprout.web.app` and never by the CLI import path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dbsprout.web.errors import WebError, WebErrorCode, raise_web_error

#: Environment variable overriding the saved-connections file location.
CONNECTIONS_PATH_ENV = "DBSPROUT_CONNECTIONS_PATH"

connections_router = APIRouter()


class SaveConnectionRequest(BaseModel):
    """Request body for ``POST /api/connections`` — a name and a connection URL.

    Both fields are validated at the boundary: a missing, blank, or
    whitespace-only ``name``/``url`` (or any unexpected field) yields FastAPI's
    ``422``. The URL is not shape-validated here — it is stored as-is after the
    password is stripped; opening a connection (and validating the URL) is a
    separate concern handled by ``POST /api/connect``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Human label for the connection.")
    url: str = Field(min_length=1, description="SQLAlchemy connection URL.")

    @field_validator("name", "url")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "must not be blank"
            raise ValueError(msg)
        return stripped


class ConnectionOut(BaseModel):
    """One saved connection in an API response — name + password-stripped URL."""

    name: str
    url: str


class ConnectionsResponse(BaseModel):
    """Envelope for ``GET /api/connections``."""

    connections: list[ConnectionOut]


class DeleteConnectionResponse(BaseModel):
    """Envelope for a successful ``DELETE /api/connections/{name}``."""

    deleted: bool


def _resolve_path() -> Path:
    """Pick the connections file: ``DBSPROUT_CONNECTIONS_PATH`` env > default."""
    env_value = os.environ.get(CONNECTIONS_PATH_ENV)
    if env_value:
        return Path(env_value)
    from dbsprout.core.connections import connections_path  # noqa: PLC0415

    return connections_path(Path.cwd())


@connections_router.get("/api/connections")
async def list_connections() -> ConnectionsResponse:
    """Return every saved connection (never a literal password)."""
    from dbsprout.core.connections import load_connections  # noqa: PLC0415

    saved = load_connections(_resolve_path())
    return ConnectionsResponse(connections=[ConnectionOut(name=c.name, url=c.url) for c in saved])


@connections_router.post("/api/connections")
async def save_connection_route(body: SaveConnectionRequest) -> ConnectionOut:
    """Persist a connection, stripping any literal password before write."""
    from dbsprout.core.connections import save_connection  # noqa: PLC0415

    saved = save_connection(_resolve_path(), body.name, body.url)
    return ConnectionOut(name=saved.name, url=saved.url)


@connections_router.delete("/api/connections/{name}")
async def delete_connection_route(request: Request, name: str) -> Any:
    """Remove a saved connection; ``404`` typed envelope when *name* is unknown."""
    from dbsprout.core.connections import delete_connection  # noqa: PLC0415

    if not delete_connection(_resolve_path(), name):
        raise_web_error(
            request,
            WebError(
                code=WebErrorCode.NOT_FOUND,
                message=f"No saved connection named {name!r}.",
                status_code=404,
                hint="List saved connections with GET /api/connections.",
            ),
        )
    return DeleteConnectionResponse(deleted=True)
