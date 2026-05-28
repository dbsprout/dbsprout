"""``POST /api/connect`` — live-database introspection endpoint (S-112).

The user submits a connection URL; the handler introspects the live database
through the S-106 core-service facade (:func:`dbsprout.core.service.load_schema`),
stores the resulting :class:`~dbsprout.schema.models.DatabaseSchema` plus the
(redacted) target into the S-111 :class:`~dbsprout.web.workspace.Workspace`
wired on ``app.state.workspace``, and returns a JSON summary (table count, table
names, dialect).

Friendly errors (FR-009 / S-116)
--------------------------------
Connection / introspection / unsupported-dialect / malformed-URL / missing-driver
failures are translated into a typed envelope via
:func:`dbsprout.web.errors.classify_connect_error` and surfaced through
:func:`dbsprout.web.errors.raise_web_error` — never a raw traceback. The handler
keeps **two** failure paths:

* the *known* exception set (driver / SQLAlchemy / ``ImportError`` /
  ``ValueError``) → the classifier returns the right code + 400; and
* an ``except Exception`` final guard that downgrades any *unexpected*
  exception to :class:`~dbsprout.web.errors.WebErrorCode.INTERNAL` (500) with
  the original exception logged at ``ERROR`` (with traceback) and a correlation
  id surfaced to the user.

Credentials are redacted in any echoed URL or message via
:func:`dbsprout.web.workspace._redact_url`, and the workspace is mutated only
on success — a failed connect can never leave half-loaded state behind.

This module is imported only by :mod:`dbsprout.web.app` (itself lazy-imported by
``dbsprout serve``); it adds nothing to the CLI import path. Heavy / CLI-adjacent
imports (the ``SchemaSource`` dataclass, SQLAlchemy, the service facade) are done
lazily inside the handler so importing the router stays cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dbsprout.web.errors import classify_connect_error, raise_web_error, web_error_internal

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.web.workspace import Workspace

connect_router = APIRouter()

# Imported lazily by the handler — kept as a module-level reference so tests can
# monkey-patch ``dbsprout.web.routers.connect.load_schema``.
load_schema: Any = None


class ConnectRequest(BaseModel):
    """Request body for ``POST /api/connect`` — a single connection URL.

    Validated at the boundary: a missing, blank, or whitespace-only ``url`` (or
    any unexpected field) yields FastAPI's ``422``. The URL *shape* is not
    pre-validated here — :func:`dbsprout.core.service.load_schema` is the single
    authority and turns a bad URL into a friendly typed envelope via the S-116
    error layer.
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


def _known_connect_exceptions() -> tuple[type[BaseException], ...]:
    """Exception types the connect path classifies as caller-actionable.

    Anything outside this tuple flows into the ``INTERNAL`` catch-all in the
    handler. SQLAlchemy's base error is appended at the seam (instead of being
    constructed inline) to keep the ``except`` clause readable and to keep
    ``sqlalchemy`` lazy-imported (the handler only imports it on first call).
    """
    import sqlalchemy as sa  # noqa: PLC0415 — lazy at first-failure call

    return (ValueError, OSError, ImportError, sa.exc.SQLAlchemyError)


@connect_router.post("/api/connect")
async def connect(request: Request, body: ConnectRequest) -> Any:
    """Introspect a live database and start a workspace session.

    On success, stores the schema + redacted target on ``app.state.workspace``
    and returns ``{"table_count", "tables", "dialect"}``. On failure, surfaces a
    typed S-116 envelope: ``4xx`` for caller-actionable errors (bad creds / bad
    URL / missing driver) and ``5xx`` only for genuine server faults (with the
    real exception logged server-side and a correlation id surfaced to the user).
    """
    from dbsprout.cli.sources import SchemaSource  # noqa: PLC0415
    from dbsprout.core.service import load_schema as _load_schema  # noqa: PLC0415
    from dbsprout.web.workspace import _redact_url  # noqa: PLC0415

    url = body.url
    redacted = _redact_url(url)
    source = SchemaSource(kind="db", raw_value=url, display_value=redacted)

    # Allow tests to monkey-patch ``dbsprout.web.routers.connect.load_schema`` —
    # they set the module-level reference; the handler honours that override
    # when present, otherwise it uses the freshly lazy-imported service facade.
    loader = load_schema or _load_schema

    try:
        schema = loader(source)
    except _known_connect_exceptions() as exc:
        return raise_web_error(request, classify_connect_error(exc, url), original=exc)
    except Exception as exc:
        return raise_web_error(request, web_error_internal(), original=exc)

    workspace = _workspace(request)
    workspace.set_schema(schema)
    workspace.set_target_url(url)
    workspace.set_source(f"db: {redacted}")
    # S-122: hydrate the workspace spec from the disk cache when a prior
    # session edited the spec for this exact schema. A miss leaves
    # ``workspace.spec`` ``None`` so ``GET /api/spec`` still triggers the
    # lazy heuristic build (existing S-118 contract preserved).
    workspace.hydrate_from_cache(schema.schema_hash())
    return _summary(schema)
