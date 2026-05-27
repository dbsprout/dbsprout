"""``POST /api/connect`` — live-database introspection endpoint (S-112).

The user submits a connection URL; the handler introspects the live database
through the S-106 core-service facade (:func:`dbsprout.core.service.load_schema`),
stores the resulting :class:`~dbsprout.schema.models.DatabaseSchema` plus the
(redacted) target into the S-111 :class:`~dbsprout.web.workspace.Workspace`
wired on ``app.state.workspace``, and returns a JSON summary (table count, table
names, dialect).

Friendly errors (FR-009)
------------------------
Connection / introspection / unsupported-dialect / malformed-URL / missing-driver
failures are translated into a clean ``4xx`` JSON body
(``{"detail": "<message>"}``) — never a raw traceback. Credentials are
**redacted** in any echoed URL or message: the raw ``user:password`` never
appears in a response. The workspace is mutated only on success, so a failed
connect can never leave half-loaded state behind.

This module is imported only by :mod:`dbsprout.web.app` (itself lazy-imported by
``dbsprout serve``); it adds nothing to the CLI import path. Heavy / CLI-adjacent
imports (the ``SchemaSource`` dataclass, SQLAlchemy, the service facade) are done
lazily inside the handler so importing the router stays cheap. Credential masking
reuses :func:`dbsprout.web.workspace._redact_url` (SQLAlchemy
``hide_password=True`` with a never-raising stdlib fallback) — the CLI scrubber is
intentionally NOT imported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.web.workspace import Workspace

connect_router = APIRouter()


class ConnectRequest(BaseModel):
    """Request body for ``POST /api/connect`` — a single connection URL.

    Validated at the boundary: a missing, blank, or whitespace-only ``url`` (or
    any unexpected field) yields FastAPI's ``422``. The URL *shape* is not
    pre-validated here — :func:`dbsprout.core.service.load_schema` is the single
    authority and turns a bad URL into a friendly ``4xx``.
    """

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, description="SQLAlchemy connection URL.")

    @field_validator("url")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "url must not be blank"
            raise ValueError(msg)
        return stripped


def _workspace(request: Request) -> Workspace:
    """Typed accessor for the shared session wired in ``app.py``."""
    return cast("Workspace", request.app.state.workspace)


def _summary(schema: DatabaseSchema) -> dict[str, Any]:
    """Shape a schema into the JSON connect summary."""
    return {
        "table_count": len(schema.tables),
        "tables": schema.table_names(),
        "dialect": schema.dialect,
    }


def _friendly_error(exc: Exception, url: str) -> str:
    """Build a clear, credential-scrubbed one-line error message.

    Most loader exceptions already embed the password-masked URL, but some
    (driver ``ImportError``, arbitrary ``ValueError``) may not — so the raw
    password and raw URL are defensively stripped from the message before it is
    returned to the client.
    """
    from dbsprout.web.workspace import _redact_url  # noqa: PLC0415

    message = str(exc) or type(exc).__name__
    redacted = _redact_url(url)
    message = message.replace(url, redacted)
    try:
        import sqlalchemy as sa  # noqa: PLC0415

        password = sa.engine.make_url(url).password
    except Exception:  # never let credential scrubbing raise
        password = None
    if password:
        message = message.replace(password, "***")
    return f"Could not connect to the database: {message}"


@connect_router.post("/api/connect")
async def connect(request: Request, body: ConnectRequest) -> dict[str, Any]:
    """Introspect a live database and start a workspace session.

    On success, stores the schema + redacted target on ``app.state.workspace``
    and returns ``{"table_count", "tables", "dialect"}``. On failure, raises a
    friendly ``400`` (no traceback, credentials redacted).
    """
    import sqlalchemy as sa  # noqa: PLC0415

    from dbsprout.cli.sources import SchemaSource  # noqa: PLC0415
    from dbsprout.core.service import load_schema  # noqa: PLC0415
    from dbsprout.web.workspace import _redact_url  # noqa: PLC0415

    url = body.url
    redacted = _redact_url(url)
    source = SchemaSource(kind="db", raw_value=url, display_value=redacted)

    try:
        schema = load_schema(source)
    except (ValueError, OSError, ImportError, sa.exc.SQLAlchemyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_friendly_error(exc, url),
        ) from None

    workspace = _workspace(request)
    workspace.set_schema(schema)
    workspace.set_target_url(url)
    workspace.set_source(f"db: {redacted}")
    return _summary(schema)
