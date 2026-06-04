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

The renderer raises an :class:`~fastapi.HTTPException` with the JSON envelope as
``detail`` — so the JSON contract (``{"detail": …}``) is the single response shape
for curl / scripted callers, the React SPA, and the existing tests. (The legacy
``HX-Request``/``error_fragment.html`` HTML branch was removed in the P1c-5
cutover along with the rest of the server-rendered UI.)

This module never imports from :mod:`dbsprout.web.app` at module level, so
siblings can import :func:`classify_connect_error` / :func:`classify_parse_error`
without circularity.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, NoReturn

from fastapi import HTTPException

if TYPE_CHECKING:
    from collections.abc import Iterable

    from starlette.requests import Request


_LOGGER = logging.getLogger("dbsprout.web.errors")


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
    # S-140 export-route guards.
    EXPORT_MULTI_TABLE_UNSUPPORTED = "EXPORT_MULTI_TABLE_UNSUPPORTED"
    EXPORT_DEPENDENCY_MISSING = "EXPORT_DEPENDENCY_MISSING"
    # S-139 update-column-route guard. Raised by ``POST /api/update-column``
    # when the workspace has no last-regenerated rows for the requested
    # ``(table, column)`` pair — the caller must run the regenerate flow
    # first so there are fresh column values to push.
    NO_REGEN = "NO_REGEN"
    # S-143 wizard step-gating guard — raised by ``POST /wizard/step/{n}``
    # when ``action=next`` is rejected because the workspace lacks the
    # artefact the next step needs (no schema, no spec, no last run, …).
    # The envelope carries ``step`` + ``missing: [...]`` as top-level extras
    # so the HTMX swap can render an actionable badge.
    STEP_GATE_BLOCKED = "STEP_GATE_BLOCKED"
    # S-145 wizard Step 3 LLM opt-in — raised by
    # ``POST /wizard/step/3/llm-spec`` when the local LLM provider cannot be
    # constructed (no ``llama-cpp-python``, no GGUF model file, …). 503
    # because the failure is server-side capability, not caller input;
    # the existing heuristic spec on the workspace stays in place.
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    # P2a-3 SSH-tunnel connect guard — raised by ``POST /api/connect`` /
    # ``/api/connect/test`` when the request carries an ``ssh`` block but the
    # optional ``[ssh]`` extra (``sshtunnel`` → ``paramiko``) is not installed.
    # 503 because the failure is a server-side capability gap, not caller
    # input — the request itself is well-formed.
    SSH_UNAVAILABLE = "SSH_UNAVAILABLE"
    # P4-9 SSH-tunnel live-failure guard — raised by ``POST /api/connect`` /
    # ``/api/connect/test`` when the bastion *connect* fails: a rejected SSH key
    # (``auth`` → 400, caller-actionable), an unreachable bastion (``host`` →
    # 502), or a remote-bind / channel failure (``forward`` → 502). The envelope
    # carries the coarse ``kind`` in ``extras`` and a credential-/target-scrubbed
    # message — never a raw 500.
    SSH_TUNNEL_FAILED = "SSH_TUNNEL_FAILED"


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
    # S-140 export-route hints.
    WebErrorCode.EXPORT_MULTI_TABLE_UNSUPPORTED: (
        "Specify a single-element `tables: [name]` subset for this format, "
        "or use the sql/json format which support multi-table exports."
    ),
    WebErrorCode.EXPORT_DEPENDENCY_MISSING: (
        "Install the missing extra and retry, e.g. pip install 'dbsprout[data]'."
    ),
    # S-139 update-column-route hint.
    WebErrorCode.NO_REGEN: (
        "POST /api/regenerate to produce fresh column values before pushing "
        "them to the target with POST /api/update-column."
    ),
    # S-143 — generic gating hint; the per-call factory always supplies a
    # more specific one when only one artefact is missing.
    WebErrorCode.STEP_GATE_BLOCKED: (
        "Complete the highlighted action on the current step before advancing."
    ),
    # S-145 — generic LLM-unavailable hint. The factory always supplies the
    # specific reason in the message; this hint nudges the user toward the
    # heuristic path that is already wired and works offline.
    WebErrorCode.LLM_UNAVAILABLE: (
        "Install an LLM provider extra (e.g. pip install 'dbsprout[llm]') or "
        "use the heuristic spec which is already populated."
    ),
    # P2a-3 — the SSH-tunnel connect path needs the optional [ssh] extra.
    WebErrorCode.SSH_UNAVAILABLE: (
        "Install the SSH-tunnel extra and retry: pip install dbsprout[ssh]."
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
# S-140 export-route factory helpers (Wave 4 — Output & Insertion).
# ---------------------------------------------------------------------------


def web_error_not_found_tables(tables: list[str]) -> WebError:
    """Surfaced when an export request lists tables absent from the last run.

    Re-uses :attr:`WebErrorCode.NOT_FOUND` (no new enum entry) — the message
    lists every offending name so the caller can correct the request without
    a round-trip.
    """
    names = ", ".join(repr(t) for t in tables)
    message = f"Tables not in the last generation result: {names}."
    return WebError(
        code=WebErrorCode.NOT_FOUND,
        message=message,
        status_code=404,
        hint=_CODE_HINTS[WebErrorCode.NOT_FOUND],
    )


def web_error_export_multi_table_unsupported(fmt: str) -> WebError:
    """The requested export format cannot be packed into a single file for a multi-table run.

    Per the S-140 brainstorm, CSV (multiple header rows) and Parquet (binary
    container) have no portable single-file representation when the scope
    resolves to more than one table; the caller must narrow the scope via
    ``tables: [name]`` or pick ``sql`` / ``json`` which support multi-table.
    """
    message = (
        f"Format {fmt!r} cannot stream multiple tables in a single file. "
        "Specify a single-element `tables: [name]` subset."
    )
    return WebError(
        code=WebErrorCode.EXPORT_MULTI_TABLE_UNSUPPORTED,
        message=message,
        status_code=422,
        hint=_CODE_HINTS[WebErrorCode.EXPORT_MULTI_TABLE_UNSUPPORTED],
    )


def web_error_no_regen() -> WebError:
    """Surfaced by ``POST /api/update-column`` when the workspace has no rows
    for the requested ``(table, column)`` pair.

    The route uses the same in-memory ``Workspace.last_result.tables_data``
    that the regenerate flow updates — so "no rows for the table" maps to
    "the user never regenerated a column there yet". Status 409 mirrors the
    rest of the workspace-state taxonomy (``NO_CONNECTION`` / ``NO_RUN`` /
    ``NO_SCHEMA``).
    """
    return WebError(
        code=WebErrorCode.NO_REGEN,
        message=("No regenerated column available; run POST /api/regenerate first."),
        status_code=409,
        hint=_CODE_HINTS[WebErrorCode.NO_REGEN],
    )


def web_error_export_dependency_missing(fmt: str, extra: str) -> WebError:
    """The writer for *fmt* needs an optional dependency that is not installed.

    Today the only path that hits this is Parquet without the ``[data]``
    extra (the writer raises ``ImportError`` when ``polars`` is absent).
    """
    message = f"Format {fmt!r} requires the optional {extra!r} extra to be installed."
    return WebError(
        code=WebErrorCode.EXPORT_DEPENDENCY_MISSING,
        message=message,
        status_code=422,
        hint=_CODE_HINTS[WebErrorCode.EXPORT_DEPENDENCY_MISSING],
    )


# ---------------------------------------------------------------------------
# S-143 wizard step-gating factory helper.
# ---------------------------------------------------------------------------


#: Per-artefact hint copy. Single source of truth so the factory below and any
#: future doc / panel can reuse the same friendly nudge per missing key.
_STEP_GATE_HINTS: dict[str, str] = {
    "schema": "Connect to a database or upload a schema file before continuing.",
    "spec": "Open Step 3 and configure at least one column before continuing.",
    "last_result": "Open Step 4 and run a generation job before continuing.",
    "validation": "Run validation on Step 5 before continuing to insert / export.",
}


def web_error_step_gate_blocked(*, step: int, missing: list[str]) -> WebError:
    """Surfaced when ``POST /wizard/step/{n}`` rejects a Next click (S-143).

    The wizard router computes ``missing`` by calling
    :meth:`dbsprout.web.wizard_state.WizardState.missing_for` and passes the
    list verbatim into ``extras``; the HTMX swap can render one badge per
    entry without re-parsing the human-readable ``message``.

    Args:
        step: The step the user tried to advance from (1..5; step 6 is the
            final step and never blocks).
        missing: Stable list of missing-artefact keys
            (``"schema"`` / ``"spec"`` / ``"last_result"`` / ``"validation"``).
            Multi-element lists are joined with " + " in the message; the
            hint defaults to the single-artefact copy when only one is
            missing, otherwise falls back to the generic gating hint.

    Returns:
        :class:`WebError` with code :class:`WebErrorCode.STEP_GATE_BLOCKED`,
        status 400, ``hint`` keyed off the single-missing artefact when
        available, and ``extras = {"step", "missing"}`` so JSON callers can
        key off the structured fields without parsing the message.
    """
    joined = " + ".join(missing) if missing else "(unknown)"
    message = (
        f"Cannot advance from step {step}: missing {joined}. "
        "Complete the highlighted action on the current step first."
    )
    if len(missing) == 1 and missing[0] in _STEP_GATE_HINTS:
        hint: str | None = _STEP_GATE_HINTS[missing[0]]
    else:
        hint = _CODE_HINTS[WebErrorCode.STEP_GATE_BLOCKED]
    return WebError(
        code=WebErrorCode.STEP_GATE_BLOCKED,
        message=message,
        status_code=400,
        hint=hint,
        extras={"step": step, "missing": list(missing)},
    )


# ---------------------------------------------------------------------------
# S-145 wizard Step 3 LLM opt-in factory helper.
# ---------------------------------------------------------------------------


def web_error_llm_unavailable(reason: str) -> WebError:
    """Surfaced by ``POST /wizard/step/3/llm-spec`` when the LLM provider fails to load.

    The wizard's Step 3 opt-in LLM path catches construction-time failures
    (``ImportError`` from ``llama-cpp-python`` missing, ``RuntimeError``
    from "no GGUF model on disk", ``OSError`` from a denied cache dir, …)
    and translates them into a 503 envelope with the original *reason*
    folded into the user-facing message.

    Status 503 because the failure is a server-side capability gap — the
    request itself is well-formed, the server simply cannot serve the
    optional LLM-driven flow. The heuristic spec that was put in place
    on entering Step 3 stays untouched on the workspace, so the user can
    keep moving with no further action required.
    """
    return WebError(
        code=WebErrorCode.LLM_UNAVAILABLE,
        message=f"LLM spec generation unavailable: {reason}",
        status_code=503,
        hint=_CODE_HINTS[WebErrorCode.LLM_UNAVAILABLE],
    )


# ---------------------------------------------------------------------------
# P2a-3 SSH-tunnel connect factory helper.
# ---------------------------------------------------------------------------


def web_error_ssh_unavailable() -> WebError:
    """Surfaced when an ``ssh`` block is supplied but the ``[ssh]`` extra is absent.

    The SSH-tunnel connect path lazy-imports ``sshtunnel`` (which pulls
    ``paramiko``) only when a request carries an ``ssh`` block — the dep is kept
    behind the optional ``[ssh]`` extra so the default install stays slim and the
    web routers import clean without it. When the lazy import fails, the connect /
    test handlers translate the ``ImportError`` into this typed 503 envelope
    (never a 500) so the user gets an actionable install hint instead of a
    traceback. The failure is well-formed-request-but-server-can't-serve, hence
    503, mirroring :func:`web_error_llm_unavailable`.

    The message intentionally carries **no** tunnel target — the bastion host and
    the remote DB address never reach the user-facing string.
    """
    return WebError(
        code=WebErrorCode.SSH_UNAVAILABLE,
        message=("SSH tunnelling is not available: the optional 'ssh' extra is not installed."),
        status_code=503,
        hint=_CODE_HINTS[WebErrorCode.SSH_UNAVAILABLE],
    )


# ---------------------------------------------------------------------------
# P4-9 SSH-tunnel live-failure factory helper.
# ---------------------------------------------------------------------------


#: Per-kind copy for a live bastion-connect failure. ``message`` carries **no**
#: target (the bastion host / remote DB address are scrubbed at the raise site in
#: :func:`dbsprout.core.ssh_tunnel.open_ssh_tunnel`); ``hint`` is the actionable
#: nudge; ``status`` is ``auth`` → 400 (caller fixes the key/user) vs
#: ``host`` / ``forward`` → 502 (the gateway hop to the DB could not be made).
_SSH_TUNNEL_FAILURE: dict[str, tuple[int, str, str]] = {
    "auth": (
        400,
        "SSH authentication to the bastion failed.",
        "Check the SSH username and that the private key at the given path is "
        "the right, unencrypted key for that bastion.",
    ),
    "host": (
        502,
        "Could not reach the SSH bastion host.",
        "Check the bastion host and port are correct and reachable from here.",
    ),
    "forward": (
        502,
        "The SSH tunnel to the database host could not be opened.",
        "The bastion was reached but the forward to the database failed; check "
        "the database host/port are reachable from the bastion.",
    ),
}


def web_error_ssh_tunnel_failed(kind: str) -> WebError:
    """Surfaced when a live bastion *connect* fails (P4-9) — never a raw 500.

    The SSH-tunnel connect path raises
    :class:`dbsprout.core.ssh_tunnel.SshTunnelConnectError` carrying a coarse
    ``kind`` (``"auth"`` / ``"host"`` / ``"forward"``) decided from the original
    exception's shape, with the bastion / remote target already scrubbed out. This
    factory turns that ``kind`` into the friendly typed envelope:

    * ``auth`` → **400** — a rejected key / wrong user is caller-actionable.
    * ``host`` → **502** — the bastion gateway itself is unreachable.
    * ``forward`` → **502** — the bastion was reached + authed but the forward to
      the database failed; this also covers any unclassified failure.

    The ``kind`` is echoed into ``extras`` so the SPA can branch (e.g. focus the
    key field on ``auth``) without re-parsing the message. An unknown ``kind``
    degrades to the ``forward`` mapping — still a typed 502, never a 500.

    The message + hint never embed the bastion host or the remote DB address.
    """
    status_code, message, hint = _SSH_TUNNEL_FAILURE.get(kind, _SSH_TUNNEL_FAILURE["forward"])
    return WebError(
        code=WebErrorCode.SSH_TUNNEL_FAILED,
        message=message,
        status_code=status_code,
        hint=hint,
        extras={"kind": kind},
    )


# ---------------------------------------------------------------------------
# Renderer: JSON error response, plus logging hook.
# ---------------------------------------------------------------------------


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
) -> NoReturn:
    """Log the failure, then raise the user-facing JSON error response.

    Logs the failure with structured context, then raises
    :class:`fastapi.HTTPException` with status ``err.status_code`` and
    ``detail = err.to_dict()``. FastAPI serialises that into the canonical
    ``{"detail": …}`` JSON envelope — the single response shape since the P1c-5
    cutover (the legacy ``HX-Request`` HTML-fragment branch was removed).
    """
    _log_error(request, err, original)
    raise HTTPException(status_code=err.status_code, detail=err.to_dict())


__all__ = [
    "WebError",
    "WebErrorCode",
    "classify_connect_error",
    "classify_parse_error",
    "raise_web_error",
    "web_error_constraint_violation",
    "web_error_empty_file",
    "web_error_export_dependency_missing",
    "web_error_export_multi_table_unsupported",
    "web_error_file_too_large",
    "web_error_internal",
    "web_error_llm_unavailable",
    "web_error_method_unsupported",
    "web_error_no_connection",
    "web_error_no_regen",
    "web_error_no_run",
    "web_error_no_schema",
    "web_error_no_spec",
    "web_error_not_found",
    "web_error_not_found_tables",
    "web_error_ssh_tunnel_failed",
    "web_error_ssh_unavailable",
    "web_error_step_gate_blocked",
    "web_error_unknown_parser",
    "web_error_write_guard_rejected",
    "web_error_write_guard_required",
]
