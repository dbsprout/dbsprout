"""Friendly user-facing error layer for the web dashboard (S-116).

Both POST endpoints under ``dbsprout/web/routers/`` (``/api/connect`` and
``/api/schema/load``) translate driver / parser exceptions into a typed
envelope before they reach the HTTP layer. The envelope shape is

.. code-block:: json

    {
      "detail": {
        "code": "AUTH_FAILED",
        "message": "Authentication failed for postgresql://user:***@host/db",
        "hint": "Double-check the username and password in the connection URL.",
        "correlation_id": "1f7c9a3e…"
      }
    }

— a closed taxonomy (:class:`WebErrorCode`), a credential-scrubbed message, an
optional fix hint, and a per-response correlation id the user can quote when
filing a bug.

Three design properties are non-negotiable here (they map 1-to-1 to the AC):

* **No tracebacks ever leak.** Routers wrap every failure path through
  :func:`raise_web_error`; ``repr(exc)`` never reaches the wire.
* **4xx is for caller-actionable errors** (bad URL / bad creds / bad file);
  ``5xx`` is reserved for genuine server faults, with code ``INTERNAL`` and the
  real exception logged at ``ERROR`` (with ``exc_info``) so server logs retain
  diagnostic value.
* **Credentials are redacted before logging.** The classifier always calls
  :func:`dbsprout.web.workspace._redact_url` before substituting a URL into a
  message, and a defensive ``replace(password, "***")`` covers driver messages
  that print the bare password.

The classifier walks the exception ``__cause__`` / ``__context__`` chain and
matches on ``type(exc).__name__`` plus a lowercase substring sweep of
``str(exc)``. This keeps the module free of driver imports (``psycopg`` /
``pymysql`` are never imported here) — drivers can be upgraded without
touching this code, and the test suite locks behaviour by raising stand-in
exceptions with the right *names*.

When the request carries ``HX-Request: true`` the renderer returns an
:class:`~starlette.responses.HTMLResponse` built from the ``error_fragment.html``
template (suitable for an ``hx-swap`` target). Without the header it raises an
:class:`~fastapi.HTTPException` with the JSON envelope as ``detail`` — so the
existing JSON contract (``{"detail": …}``) is preserved for curl / scripted
callers and the existing tests.

This module never imports from :mod:`dbsprout.web.app` at module level (only
the renderer's ``request.app.state.templates`` access pulls in templates lazily
at call time), so siblings can import :func:`classify_connect_error` /
:func:`classify_parse_error` without circularity.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, cast

from fastapi import HTTPException

if TYPE_CHECKING:
    from collections.abc import Iterable

    from starlette.requests import Request
    from starlette.responses import HTMLResponse


_LOGGER = logging.getLogger("dbsprout.web.errors")

#: Header that marks an HTMX-driven request. When present, the renderer returns
#: an HTML fragment instead of raising the JSON-shaped ``HTTPException``.
_HTMX_HEADER = "hx-request"


class WebErrorCode(str, Enum):
    """Closed taxonomy of friendly error codes.

    The set is closed by design — front-end code keys off these strings and the
    test suite locks the full membership in
    :func:`tests.test_web.test_errors.test_web_error_code_enum_is_closed_set`.
    Adding or renaming a code requires a coordinated PRD update.

    We subclass ``str`` rather than use :class:`enum.StrEnum` directly because
    the project's mypy target is ``py310`` (StrEnum landed in 3.11); the
    ``str, Enum`` mixin gives identical JSON-friendly behaviour without
    requiring a target-version bump.
    """

    CONN_REFUSED = "CONN_REFUSED"
    AUTH_FAILED = "AUTH_FAILED"
    UNKNOWN_DIALECT = "UNKNOWN_DIALECT"
    MISSING_DRIVER = "MISSING_DRIVER"
    MALFORMED_URL = "MALFORMED_URL"
    PARSE_ERROR = "PARSE_ERROR"
    EMPTY_FILE = "EMPTY_FILE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNKNOWN_PARSER = "UNKNOWN_PARSER"
    INTERNAL = "INTERNAL"
    # S-136 insert-route guards (Wave 1 of Output & Insertion).
    NO_CONNECTION = "NO_CONNECTION"
    NO_RUN = "NO_RUN"
    # S-136 forward-handoff to S-137 (write-guard confirmation token).
    WRITE_GUARD_REQUIRED = "WRITE_GUARD_REQUIRED"
    # S-137: token present but failed HMAC / scope / TTL / single-use check.
    WRITE_GUARD_REJECTED = "WRITE_GUARD_REJECTED"
    # S-131 regenerate-route guards (Wave 3 — Granular Control).
    NO_SCHEMA = "NO_SCHEMA"
    NO_SPEC = "NO_SPEC"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    NOT_FOUND = "NOT_FOUND"
    # S-141 insert-route method select (auto/batch/copy). Raised when the
    # caller pins ``method="copy"`` against a dialect that has no COPY /
    # LOAD DATA equivalent (sqlite, mssql, oracle, …) or when the optional
    # driver for COPY is not installed (psycopg / pymysql).
    METHOD_UNSUPPORTED = "METHOD_UNSUPPORTED"


@dataclass(frozen=True)
class WebError:
    """A single user-facing error description.

    The class is frozen — every classifier returns a fresh instance, and the
    correlation id is generated at construction time so callers cannot
    accidentally share one across responses.

    ``extras`` carries code-specific structured fields that need to land
    *at the top of* the envelope payload alongside ``code`` / ``message``
    (e.g. S-141's ``METHOD_UNSUPPORTED`` carries ``dialect`` / ``method`` /
    ``supported`` so the Studio JS can render an actionable picker without
    re-parsing the message). The default empty dict keeps the envelope
    backward-compatible with every existing factory.
    """

    code: WebErrorCode
    message: str
    status_code: int
    hint: str | None = None
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    extras: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Render the envelope payload, omitting an unset ``hint``.

        Code-specific ``extras`` are merged at the top of the payload —
        existing callers that pass no extras see the historical shape
        (``code`` / ``message`` / ``correlation_id`` [/ ``hint``]) unchanged.
        """
        payload: dict[str, object] = {
            "code": self.code.value,
            "message": self.message,
            "correlation_id": self.correlation_id,
        }
        if self.hint is not None:
            payload["hint"] = self.hint
        # Merge extras last so a malicious factory can't accidentally
        # override the closed ``code`` / ``correlation_id`` keys — those
        # are stamped first, ``extras`` simply adds siblings. We DO allow
        # overriding ``message`` / ``hint`` if a factory explicitly wants
        # to render a structured message inline (none do today).
        for key, value in self.extras.items():
            if key in {"code", "correlation_id"}:
                continue
            payload[key] = value
        return payload


# ---------------------------------------------------------------------------
# Credential scrubbing helpers.
# ---------------------------------------------------------------------------


def _redact(url: str | None) -> str:
    """Return a credential-scrubbed display version of *url* (never raises)."""
    if not url:
        return ""
    from dbsprout.web.workspace import _redact_url  # noqa: PLC0415 — lazy import

    return _redact_url(url)


def _extract_password(url: str | None) -> str | None:
    """Best-effort password extraction for defensive substitution.

    Some driver messages embed the bare password (rather than the full URL),
    so the classifier needs to scrub it directly. Failure to parse the URL is
    swallowed — the caller is already on the failure path.
    """
    if not url:
        return None
    try:
        import sqlalchemy as sa  # noqa: PLC0415 — already a project dep

        password = sa.engine.make_url(url).password
    except Exception:
        return None
    return password


def _scrub_message(raw: str, url: str | None) -> str:
    """Strip the raw URL and password from a driver message."""
    redacted = _redact(url)
    scrubbed = raw
    if url:
        scrubbed = scrubbed.replace(url, redacted)
    password = _extract_password(url)
    if password:
        scrubbed = scrubbed.replace(password, "***")
    return scrubbed


# ---------------------------------------------------------------------------
# Exception chain walking — name + message-based classifier.
# ---------------------------------------------------------------------------


def _walk_chain(exc: BaseException) -> Iterable[BaseException]:
    """Yield ``exc`` followed by its ``__cause__`` / ``__context__`` chain."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _matches(text: str, patterns: Iterable[str]) -> bool:
    """Case-insensitive substring match against any of *patterns*."""
    lowered = text.lower()
    return any(p in lowered for p in patterns)


# Patterns matched against ``str(exc)`` (case-insensitive). Order matters
# only when two codes could match the same exception — currently they don't.
_CONNECT_PATTERNS: dict[WebErrorCode, tuple[str, ...]] = {
    WebErrorCode.AUTH_FAILED: (
        "password authentication failed",
        "authentication failed",
        "access denied for user",
        "role does not exist",
        "fatal:  password",
    ),
    WebErrorCode.UNKNOWN_DIALECT: (
        "unsupported dialect",
        "can't load plugin: sqlalchemy.dialects",
        "no dialect",
    ),
    WebErrorCode.MALFORMED_URL: (
        "could not parse",
        "invalid url",
        "rfc1738",
    ),
    WebErrorCode.CONN_REFUSED: (
        "connection refused",
        "could not connect",
        "could not translate host",
        "name or service not known",
        "name resolution failure",
        "no route to host",
        "host is down",
        "is the server running",
    ),
}

# Type-name → code map. We never import the driver modules; the *name* alone
# is enough to dispatch.
_CONNECT_TYPE_NAMES: dict[str, WebErrorCode] = {
    "NoSuchModuleError": WebErrorCode.UNKNOWN_DIALECT,
    "ArgumentError": WebErrorCode.MALFORMED_URL,
    "ImportError": WebErrorCode.MISSING_DRIVER,
    "ModuleNotFoundError": WebErrorCode.MISSING_DRIVER,
}


_CODE_HINTS: dict[WebErrorCode, str] = {
    WebErrorCode.CONN_REFUSED: ("Check the host, port, and that the database server is reachable."),
    WebErrorCode.AUTH_FAILED: ("Double-check the username and password in the connection URL."),
    WebErrorCode.UNKNOWN_DIALECT: (
        "Use one of the supported dialects (postgresql, mysql, sqlite, mssql)."
    ),
    WebErrorCode.MALFORMED_URL: ("Use the format dialect+driver://user:password@host:port/dbname."),
    WebErrorCode.PARSE_ERROR: (
        "Open the file in your editor and fix the reported line, then upload again."
    ),
    WebErrorCode.EMPTY_FILE: "Upload a non-empty schema file.",
    WebErrorCode.FILE_TOO_LARGE: "Split or compress the schema, then upload again.",
    # S-136 hints.
    WebErrorCode.NO_CONNECTION: "POST /api/connect with a target database URL first.",
    WebErrorCode.NO_RUN: "POST /api/generate to produce data before inserting.",
    WebErrorCode.WRITE_GUARD_REQUIRED: (
        "Request a confirmation token from POST /api/insert/preview (S-137) and resubmit."
    ),
    WebErrorCode.WRITE_GUARD_REJECTED: (
        "Re-fetch a token from POST /api/insert/preview — yours expired, was "
        "already used, or was bound to a different target/scope."
    ),
    # S-131 regenerate-route hints.
    WebErrorCode.NO_SCHEMA: ("Load a schema first via POST /api/connect or POST /api/schema/load."),
    WebErrorCode.NO_SPEC: (
        "Build or load a DataSpec first; the regenerate path needs a spec when engine='spec'."
    ),
    WebErrorCode.CONSTRAINT_VIOLATION: (
        "Pick a non-PK, non-FK-referenced column; regenerating these would "
        "violate referential integrity."
    ),
    WebErrorCode.NOT_FOUND: (
        "Check the schema and re-issue the request with a valid table / column name."
    ),
    # S-141 default hint — most callers will pass a more specific hint
    # explaining *why* the method is unsupported (wrong dialect vs. missing
    # driver). This generic fallback is correct for the wrong-dialect case.
    WebErrorCode.METHOD_UNSUPPORTED: (
        "Pick one of the supported methods listed in 'supported' (commonly 'auto' or 'batch')."
    ),
}


def _missing_driver_hint(message: str) -> str:
    """Build a ``pip install`` hint from the module name in *message*."""
    match = re.search(r"['\"]([A-Za-z0-9_.\-]+)['\"]", message)
    package = match.group(1) if match else "the missing driver"
    base = f"Install the database driver, e.g. pip install {package}"
    return f"{base}, and try again."


def classify_connect_error(exc: BaseException, url: str) -> WebError:
    """Translate a live-database exception into a typed :class:`WebError`.

    Walks ``exc.__cause__`` / ``__context__`` looking for the first frame whose
    type name or message matches a known code. Falls back to :class:`WebErrorCode.CONN_REFUSED`
    for un-classified ``ValueError`` / ``sqlalchemy`` errors (those *are*
    caller-actionable), and to :class:`WebErrorCode.INTERNAL` (500) for
    genuinely unexpected exception types like ``RuntimeError``.
    """
    redacted = _redact(url)
    for frame in _walk_chain(exc):
        type_name = type(frame).__name__
        message = str(frame) or type_name
        # 1. Type-name dispatch (exact match wins).
        if type_name in _CONNECT_TYPE_NAMES:
            code = _CONNECT_TYPE_NAMES[type_name]
            if code is WebErrorCode.MISSING_DRIVER:
                return WebError(
                    code=code,
                    message=f"Database driver is not installed: {_scrub_message(message, url)}",
                    status_code=400,
                    hint=_missing_driver_hint(message),
                )
            scrubbed = _scrub_message(message, url)
            return WebError(
                code=code,
                message=f"{_code_prefix(code, redacted)}: {scrubbed}",
                status_code=400,
                hint=_CODE_HINTS.get(code),
            )
        # 2. Message keyword dispatch.
        for code, patterns in _CONNECT_PATTERNS.items():
            if _matches(message, patterns):
                scrubbed = _scrub_message(message, url)
                return WebError(
                    code=code,
                    message=f"{_code_prefix(code, redacted)}: {scrubbed}",
                    status_code=400,
                    hint=_CODE_HINTS.get(code),
                )

    # 3. Fallbacks.
    type_name = type(exc).__name__
    if isinstance(exc, ValueError):
        scrubbed = _scrub_message(str(exc) or type_name, url)
        return WebError(
            code=WebErrorCode.CONN_REFUSED,
            message=f"{_code_prefix(WebErrorCode.CONN_REFUSED, redacted)}: {scrubbed}",
            status_code=400,
            hint=_CODE_HINTS[WebErrorCode.CONN_REFUSED],
        )
    # SQLAlchemy errors that didn't pattern-match: still caller-actionable.
    if type_name.endswith("Error") and type_name.startswith(
        ("Operational", "Interface", "Database", "Programming", "Internal")
    ):
        scrubbed = _scrub_message(str(exc) or type_name, url)
        return WebError(
            code=WebErrorCode.CONN_REFUSED,
            message=f"{_code_prefix(WebErrorCode.CONN_REFUSED, redacted)}: {scrubbed}",
            status_code=400,
            hint=_CODE_HINTS[WebErrorCode.CONN_REFUSED],
        )
    return web_error_internal()


def _code_prefix(code: WebErrorCode, redacted: str) -> str:
    """Human-readable lead-in per code."""
    if code is WebErrorCode.AUTH_FAILED:
        return f"Authentication failed for {redacted}"
    if code is WebErrorCode.CONN_REFUSED:
        return f"Could not connect to {redacted}"
    if code is WebErrorCode.UNKNOWN_DIALECT:
        return "The database dialect is not supported"
    if code is WebErrorCode.MALFORMED_URL:
        return "The connection URL is not valid"
    return str(code.value)


# ---------------------------------------------------------------------------
# Parse-side classifier + factory helpers.
# ---------------------------------------------------------------------------


def classify_parse_error(exc: BaseException, filename: str | None) -> WebError:
    """Translate a schema-parser exception into a typed :class:`WebError`."""
    name = (filename or "<upload>").rsplit("/", 1)[-1]
    if isinstance(exc, ValueError | OSError):
        return WebError(
            code=WebErrorCode.PARSE_ERROR,
            message=f"Could not parse {name}: {exc}",
            status_code=400,
            hint=_CODE_HINTS[WebErrorCode.PARSE_ERROR],
        )
    return web_error_internal()


def web_error_empty_file(filename: str | None) -> WebError:
    name = (filename or "<upload>").rsplit("/", 1)[-1]
    return WebError(
        code=WebErrorCode.EMPTY_FILE,
        message=f"Uploaded file {name} is empty.",
        status_code=400,
        hint=_CODE_HINTS[WebErrorCode.EMPTY_FILE],
    )


def web_error_file_too_large(limit_bytes: int) -> WebError:
    mb = max(1, limit_bytes // (1024 * 1024))
    return WebError(
        code=WebErrorCode.FILE_TOO_LARGE,
        message=f"Uploaded file exceeds the {mb} MB limit.",
        status_code=413,
        hint=_CODE_HINTS[WebErrorCode.FILE_TOO_LARGE],
    )


def web_error_unknown_parser(parser: str, allowed: Iterable[str]) -> WebError:
    allowed_list = ", ".join(sorted(allowed))
    return WebError(
        code=WebErrorCode.UNKNOWN_PARSER,
        message=f"Unknown parser {parser!r}. Supported parsers: {allowed_list}.",
        status_code=400,
        hint="Pick one of the supported parsers and resubmit.",
    )


def web_error_internal() -> WebError:
    """Generic catch-all — no hint, no exception text, INTERNAL/500."""
    return WebError(
        code=WebErrorCode.INTERNAL,
        message="Unexpected error",
        status_code=500,
    )


# ---------------------------------------------------------------------------
# S-136 insert-route factory helpers (Wave 1 of Output & Insertion).
# ---------------------------------------------------------------------------


def web_error_no_connection() -> WebError:
    """No target DB is wired on the workspace yet — 409, caller-actionable."""
    return WebError(
        code=WebErrorCode.NO_CONNECTION,
        message="No target database is connected; POST /api/connect first.",
        status_code=409,
        hint=_CODE_HINTS[WebErrorCode.NO_CONNECTION],
    )


def web_error_no_run() -> WebError:
    """No generation result is available on the workspace yet — 409."""
    return WebError(
        code=WebErrorCode.NO_RUN,
        message="No generation result available; POST /api/generate first.",
        status_code=409,
        hint=_CODE_HINTS[WebErrorCode.NO_RUN],
    )


def web_error_write_guard_required() -> WebError:
    """The write path is closed unless the request carries a confirmation token.

    S-137 (Wave 2) lands the real HMAC verification + scope binding; for S-136
    the gate is in place from day one so the production path can never reach
    a live INSERT without going through it.
    """
    return WebError(
        code=WebErrorCode.WRITE_GUARD_REQUIRED,
        message="A confirmation token is required to insert into the target database.",
        status_code=403,
        hint=_CODE_HINTS[WebErrorCode.WRITE_GUARD_REQUIRED],
    )


def web_error_write_guard_rejected() -> WebError:
    """The provided confirmation token failed HMAC / scope / TTL / single-use check (S-137)."""
    return WebError(
        code=WebErrorCode.WRITE_GUARD_REJECTED,
        message=(
            "Confirmation token rejected: expired, single-use already consumed, "
            "or bound to a different target / scope."
        ),
        status_code=403,
        hint=_CODE_HINTS[WebErrorCode.WRITE_GUARD_REJECTED],
    )


# ---------------------------------------------------------------------------
# S-131 regenerate-route factory helpers (Wave 3 — Granular Control).
# ---------------------------------------------------------------------------


def web_error_no_schema() -> WebError:
    """No schema is loaded on the workspace yet — 409, caller-actionable."""
    return WebError(
        code=WebErrorCode.NO_SCHEMA,
        message="No schema loaded; connect to a database or upload a schema first.",
        status_code=409,
        hint=_CODE_HINTS[WebErrorCode.NO_SCHEMA],
    )


def web_error_no_spec() -> WebError:
    """No DataSpec is available on the workspace yet — 409."""
    return WebError(
        code=WebErrorCode.NO_SPEC,
        message="No DataSpec available; build a spec before requesting a spec-driven regen.",
        status_code=409,
        hint=_CODE_HINTS[WebErrorCode.NO_SPEC],
    )


def web_error_constraint_violation(
    *,
    table: str,
    column: str | None,
    reason: str,
) -> WebError:
    """Surfaced when a regenerate call hits a referential-integrity invariant.

    The two triggering reasons are ``primary_key`` (the column is part of the
    table's PK, which the regen path preserves byte-identically) and
    ``fk_referenced`` (the column is referenced by another table's FK, so
    re-rolling would orphan child rows). The reason is echoed verbatim into
    the user-facing message + the envelope so the Studio UI can render an
    actionable badge next to the offending cell.
    """
    column_part = f".{column}" if column else ""
    message = (
        f"Cannot regenerate {table}{column_part}: {reason.replace('_', ' ')}. "
        "Pick a non-PK, non-FK-referenced column."
    )
    return WebError(
        code=WebErrorCode.CONSTRAINT_VIOLATION,
        message=message,
        status_code=409,
        hint=_CODE_HINTS[WebErrorCode.CONSTRAINT_VIOLATION],
    )


def web_error_not_found(*, table: str, column: str | None = None) -> WebError:
    """Surfaced when the table or column referenced by a request is unknown."""
    if column is not None:
        message = f"Column {table}.{column!r} is not in the loaded schema."
    else:
        message = f"Table {table!r} is not in the loaded schema."
    return WebError(
        code=WebErrorCode.NOT_FOUND,
        message=message,
        status_code=404,
        hint=_CODE_HINTS[WebErrorCode.NOT_FOUND],
    )


# ---------------------------------------------------------------------------
# S-141 insert-method-select factory helper.
# ---------------------------------------------------------------------------


def web_error_method_unsupported(
    *,
    dialect: str,
    method: str,
    supported: list[str],
    hint: str | None = None,
) -> WebError:
    """Surfaced when the caller pins a *method* the *dialect* cannot serve.

    Two trigger paths:

    1. **Wrong dialect** — e.g. ``method="copy"`` against sqlite / mssql /
       oracle (no COPY equivalent). The default hint is fine here.
    2. **Missing optional driver** — e.g. ``method="copy"`` against PG with
       ``psycopg`` not installed (or MySQL with ``pymysql`` missing). The
       caller passes a tailored *hint* mentioning the pip install.

    The envelope carries ``dialect`` / ``method`` / ``supported`` at the top
    of the payload (alongside ``code``) so the Studio JS can render a
    self-contained "this method isn't available here, pick one of: …"
    message without re-parsing the human-readable ``message``.
    """
    supported_list = ", ".join(supported) or "(none)"
    message = (
        f"Method {method!r} is not supported for dialect {dialect!r}. "
        f"Supported methods: {supported_list}."
    )
    return WebError(
        code=WebErrorCode.METHOD_UNSUPPORTED,
        message=message,
        status_code=409,
        hint=hint if hint is not None else _CODE_HINTS[WebErrorCode.METHOD_UNSUPPORTED],
        extras={
            "dialect": dialect,
            "method": method,
            "supported": list(supported),
        },
    )


# ---------------------------------------------------------------------------
# Renderer: HTMX-aware response, plus logging hook.
# ---------------------------------------------------------------------------


def _wants_htmx(request: Request) -> bool:
    """Return ``True`` if the request carries ``HX-Request: true``."""
    value = request.headers.get(_HTMX_HEADER)
    return value is not None and value.lower() == "true"


def _log_error(
    request: Request,
    err: WebError,
    original: BaseException | None,
) -> None:
    """Log the failure once with structured context.

    ``INTERNAL`` is logged at ``ERROR`` with ``exc_info`` so the traceback hits
    the server log while staying out of the user-facing body. Known codes log
    at ``WARNING`` without traceback — they are caller-actionable.
    """
    extra = {
        "code": err.code.value,
        "status": err.status_code,
        "correlation_id": err.correlation_id,
        "route": request.url.path,
        "exc_type": type(original).__name__ if original is not None else None,
    }
    message = f"web-error {err.code.value} {err.correlation_id}"
    if err.code is WebErrorCode.INTERNAL:
        _LOGGER.error(message, extra=extra, exc_info=original)
    else:
        _LOGGER.warning(message, extra=extra)


def raise_web_error(
    request: Request,
    err: WebError,
    *,
    original: BaseException | None = None,
) -> HTMLResponse:
    """Log + raise/return the user-facing error response.

    Behaviour depends on the ``HX-Request`` header on *request*:

    * **No header** — logs the failure, then raises :class:`fastapi.HTTPException`
      with status ``err.status_code`` and ``detail = err.to_dict()``. FastAPI
      serialises that into the canonical ``{"detail": …}`` JSON envelope.
    * **Header set** — logs the failure and returns an :class:`HTMLResponse`
      rendered from ``error_fragment.html``. Callers using this branch are
      responsible for returning the response from their handler (raising would
      bypass the HTMX swap target).
    """
    _log_error(request, err, original)
    if _wants_htmx(request):
        from fastapi.templating import (  # noqa: PLC0415, TC002 — lazy + needed at runtime
            Jinja2Templates,
        )

        templates = cast("Jinja2Templates", request.app.state.templates)
        return templates.TemplateResponse(
            request,
            "error_fragment.html",
            {"error": err.to_dict(), "status_code": err.status_code},
            status_code=err.status_code,
        )
    raise HTTPException(status_code=err.status_code, detail=err.to_dict())


__all__ = [
    "WebError",
    "WebErrorCode",
    "classify_connect_error",
    "classify_parse_error",
    "raise_web_error",
    "web_error_constraint_violation",
    "web_error_empty_file",
    "web_error_file_too_large",
    "web_error_internal",
    "web_error_method_unsupported",
    "web_error_no_connection",
    "web_error_no_run",
    "web_error_no_schema",
    "web_error_no_spec",
    "web_error_not_found",
    "web_error_unknown_parser",
    "web_error_write_guard_rejected",
    "web_error_write_guard_required",
]
