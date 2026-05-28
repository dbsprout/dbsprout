"""``POST /api/insert`` — submit a dialect-aware insertion run as a job (S-136).

The Studio user clicks **Insert** in the dashboard; the handler validates the
request body (``{tables?: [str] | null, confirmation_token?: str | null}``),
reads the loaded :class:`~dbsprout.web.workspace.Workspace` (S-111, wired on
``app.state.workspace``), resolves the FK-safe scope from
``workspace.get_last_result().insertion_order``, selects the right
dialect-aware writer from :mod:`dbsprout.output` (PG COPY · MySQL LOAD DATA ·
SaBatch fallback — the same policy the CLI uses in
``dbsprout/cli/commands/generate.py::_run_direct_insert``), and submits the
insert as a background job via the :class:`~dbsprout.web.jobs.JobManager`
(S-108). Live per-table progress streams over the existing
``/ws/jobs/{job_id}`` (S-109) thanks to the closure emitting
:class:`~dbsprout.generate.progress.ProgressEvent`\\ s around each writer call.

Module structure (S-137 forward-handoff)
----------------------------------------
The module is split into two clearly demarcated regions so the **Wave 2**
story (S-137 — *write-guard confirmation token*) can land in the SAME file
without touching the insert handler:

* ``# region: write-guard (S-137)`` … ``# endregion`` — currently a small
  set of stubs (``_require_confirmation_token`` + ``_validate_confirmation_token``).
  When S-137 lands, the stub body of ``_validate_confirmation_token`` becomes
  a real HMAC verification + scope binding; ``_require_confirmation_token``
  already enforces ``403 WRITE_GUARD_REQUIRED`` when the token is missing,
  so the write path is closed from day one (S-136).
* ``# region: POST /api/insert (S-136)`` … ``# endregion`` — the insert
  handler itself + its private helpers (writer dispatch, scope resolution,
  job closure, credential scrub).

Guards (all return typed envelopes via :mod:`dbsprout.web.errors`)
------------------------------------------------------------------
* No ``confirmation_token`` ⇒ ``403 WRITE_GUARD_REQUIRED``.
* No target wired on the workspace ⇒ ``409 NO_CONNECTION``.
* No generation result on the workspace ⇒ ``409 NO_RUN``.
* Unknown table in ``tables[]`` ⇒ ``422`` (parameter-side).
* Second concurrent submit ⇒ ``409`` (single-active model — S-108).

Credential redaction (DBS-139 forward note)
-------------------------------------------
The writer DOES open a live DB connection — credential leaks ARE a real
concern. The closure reads the raw target URL via
``workspace.peek_target_url()`` (closure-captured), wraps each writer call
in ``try/except``, and on failure scrubs the workspace's raw target + bare
password from the message using :func:`dbsprout.web.workspace._redact_url`
+ a SQLAlchemy password replace before re-raising a fresh exception. The
manager records ``str(exc)`` into ``JobRecord.error``; the raw
``user:password`` never reaches the API response.

Lazy-import contract
--------------------
``dbsprout serve`` lazy-imports the web layer; importing
``dbsprout.cli.app`` must never pull FastAPI or the generation pipeline.
Accordingly this module imports only stdlib + FastAPI (which is already
gated by the ``[web]`` extra) at module level. The writers, the
``ProgressEvent`` model, the workspace redactor, and the ``GenerationCancelled``
control exception are all imported lazily inside the closure / handler.

Module owns its own :class:`~fastapi.APIRouter` (``insert_router``),
registered by :func:`dbsprout.web.app.create_app` inside a delimited region.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import TYPE_CHECKING, Any, Literal, cast

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Callable

    from dbsprout.generate.orchestrator import GenerateResult
    from dbsprout.generate.progress import CancelToken, ProgressEvent
    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.web.workspace import Workspace


insert_router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────
# region: write-guard (S-137)
# ─────────────────────────────────────────────────────────────────────────
# S-137 lands the real HMAC verification + scope binding plus the
# ``POST /api/insert/preview`` endpoint that issues short-lived,
# single-use, scope-bound tokens.
#
# Design:
#
#   * The token is ``base64url(json(payload)).base64url(hmac_sha256(secret,
#     json(payload)))``. ``payload`` carries ONLY hashes — never the raw
#     DSN — so a leaked token never leaks credentials.
#   * The HMAC secret comes from ``app.state.config.web.secret_key`` if
#     present, else a 32-byte secret is generated lazily at the first
#     verification call and stashed on ``app.state.write_guard_secret``.
#     The secret is never persisted to disk.
#   * ``app.state.write_guard_issued`` is a ``dict[nonce -> exp]`` of
#     unconsumed tokens — single-use enforcement pops the entry; expired
#     entries are garbage-collected lazily on every preview call.
#   * The test-only env var ``DBSPROUT_DISABLE_WRITE_GUARD`` short-circuits
#     ONLY the *missing-token* path (preserving the S-136 contract for the
#     existing happy-path tests in ``test_insert.py``). A present-but-
#     invalid token is ALWAYS rejected, env var or no env var — production
#     code path cannot be bypassed by a non-local actor.
# ─────────────────────────────────────────────────────────────────────────

#: Test-only escape hatch env var. Production callers never set this.
_WRITE_GUARD_DISABLED_ENV = "DBSPROUT_DISABLE_WRITE_GUARD"

#: Preview token TTL in seconds (5 min default — long enough for a human
#: to read the modal and click Confirm; short enough to bound replay).
_PREVIEW_TOKEN_TTL = 300

#: HMAC secret size — stdlib :func:`secrets.token_bytes` produces a
#: cryptographically strong random byte-string of this length.
_SECRET_BYTES = 32


def _write_guard_disabled() -> bool:
    """``True`` when the test-only escape hatch env var is set to a truthy value.

    The env var ONLY short-circuits the *missing-token* path (handled in
    :func:`_require_confirmation_token` below). It has no effect on the
    HMAC verification — a present-but-invalid token is always rejected.
    """
    raw = os.environ.get(_WRITE_GUARD_DISABLED_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _b64url_encode(raw: bytes) -> str:
    """Padding-free base64url encode."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    """Padding-free base64url decode (raises ``ValueError`` on malformed input)."""
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def _get_write_guard_secret(app: FastAPI) -> bytes:
    """Return the per-app HMAC secret, generating + memoising one on first call.

    Resolution order:
      1. ``app.state.config.web.secret_key`` — if a config object is wired
         (production / when ``--secret-key`` is provided), use it directly.
      2. ``app.state.write_guard_secret`` — once we generate a random
         secret on first call, memoise it on ``app.state`` so subsequent
         calls within the same app instance see the same secret. The
         secret is NEVER persisted to disk.

    The function is idempotent and side-effect-free except for memoising
    a freshly-generated secret on the app state.
    """
    config = getattr(app.state, "config", None)
    if config is not None:
        web_config = getattr(config, "web", None)
        if web_config is not None:
            secret_key = getattr(web_config, "secret_key", None)
            if secret_key:
                # Normalise str → bytes if the user supplied a string.
                if isinstance(secret_key, str):
                    return secret_key.encode("utf-8")
                return cast("bytes", secret_key)
    existing = getattr(app.state, "write_guard_secret", None)
    if existing is not None:
        return cast("bytes", existing)
    fresh = secrets.token_bytes(_SECRET_BYTES)
    app.state.write_guard_secret = fresh
    return fresh


def _hash_target(target_url: str) -> str:
    """Return a stable SHA-256 hex digest of *target_url*.

    Used in the token payload so the server can re-derive the binding on
    the wire without storing the raw DSN (no credential leak via the
    token).
    """
    return hashlib.sha256(target_url.encode("utf-8")).hexdigest()


def _hash_scope(scope_pairs: list[tuple[str, int]]) -> str:
    """Return a stable SHA-256 hex digest of the (table, row_count) scope.

    Ordering is normalised before hashing so that two equivalent scopes
    presented in different orders yield the same digest. The pair-form
    binds the row count too, so a regen that produces a different row
    count for the same table invalidates the token (the user gets a
    fresh preview).
    """
    canonical = "\n".join(f"{name}:{count}" for name, count in sorted(scope_pairs))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _encode_token(payload: dict[str, Any], secret: bytes) -> str:
    """Encode *payload* into a signed ``payload_b64.sig_b64`` token string."""
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(secret, payload_bytes, hashlib.sha256).digest()
    return f"{_b64url_encode(payload_bytes)}.{_b64url_encode(sig)}"


def _decode_token(token: str, secret: bytes) -> dict[str, Any] | None:
    """Verify HMAC + decode payload. Returns ``None`` on ANY failure.

    Constant-time HMAC compare via :func:`hmac.compare_digest`. We never
    raise here — callers want a boolean / ``None`` result so they can
    surface a uniform 403 rather than leaking exception type info.
    """
    if not isinstance(token, str) or "." not in token:
        return None
    try:
        payload_b64, sig_b64 = token.split(".", 1)
        payload_bytes = _b64url_decode(payload_b64)
        signature = _b64url_decode(sig_b64)
    except (ValueError, TypeError):
        return None
    expected = hmac.new(secret, payload_bytes, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        decoded = json.loads(payload_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    return cast("dict[str, Any]", decoded)


def _gc_issued_tokens(issued: dict[str, int]) -> None:
    """Remove expired nonces from the issued-token registry (in-place)."""
    now = int(time.time())
    expired = [nonce for nonce, exp in issued.items() if exp <= now]
    for nonce in expired:
        issued.pop(nonce, None)


def _get_issued_registry(app: FastAPI) -> dict[str, int]:
    """Return the per-app issued-token registry, creating it on first call."""
    registry = getattr(app.state, "write_guard_issued", None)
    if registry is None:
        registry = {}
        app.state.write_guard_issued = registry
    return cast("dict[str, int]", registry)


def _validate_confirmation_token(  # noqa: PLR0911 — each guard returns False
    token: str,
    *,
    scope: list[str],
    target_url: str,
    app: FastAPI | None = None,
    row_counts: dict[str, int] | None = None,
) -> bool:
    """Verify *token* against the announced *scope* + *target_url*.

    Returns ``True`` only if every check passes:

      1. Token decodes + HMAC verifies under the per-app secret.
      2. Token has not expired (``exp > now``).
      3. ``target_hash`` matches ``_hash_target(target_url)``.
      4. ``scope_hash`` matches ``_hash_scope(zip(scope, row_counts))``.
      5. The ``nonce`` is in the issued-token registry (single-use).

    On success the nonce is popped from the registry so a second call
    with the same token fails (single-use).

    Any failure path returns ``False`` — callers translate to
    ``403 WRITE_GUARD_REJECTED``.

    The ``app`` parameter is optional only for the historical S-136
    signature compatibility; when ``None`` the function returns ``False``
    (we cannot verify without access to the per-app secret + registry).
    """
    if app is None or not token:
        return False
    secret = _get_write_guard_secret(app)
    payload = _decode_token(token, secret)
    if payload is None:
        return False
    exp = payload.get("exp")
    nonce = payload.get("nonce")
    claimed_target = payload.get("target_hash")
    claimed_scope = payload.get("scope_hash")
    if not isinstance(exp, int):
        return False
    if exp <= int(time.time()):
        return False
    if not isinstance(nonce, str):
        return False
    if not isinstance(claimed_target, str) or not isinstance(claimed_scope, str):
        return False
    if not hmac.compare_digest(claimed_target, _hash_target(target_url)):
        return False
    rc = row_counts or {}
    expected_scope_hash = _hash_scope([(t, int(rc.get(t, 0))) for t in scope])
    if not hmac.compare_digest(claimed_scope, expected_scope_hash):
        return False
    issued = _get_issued_registry(app)
    if nonce not in issued:
        return False
    issued.pop(nonce, None)
    return True


def _require_confirmation_token(
    token: str | None,
    *,
    scope: list[str],
    target_url: str,
    app: FastAPI | None = None,
    row_counts: dict[str, int] | None = None,
) -> None:
    """Enforce the write-guard gate.

    * Token missing AND env var unset → 403 ``WRITE_GUARD_REQUIRED``.
    * Token missing AND env var set → returns (test-only escape hatch).
    * Token present BUT fails HMAC / scope / TTL / single-use → 403
      ``WRITE_GUARD_REJECTED`` regardless of the env var (production
      code path is never bypassable for present-but-invalid tokens).
    """
    if not token:
        if _write_guard_disabled():
            return
        from dbsprout.web.errors import (  # noqa: PLC0415
            web_error_write_guard_required,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=web_error_write_guard_required().to_dict(),
        )
    # Token present — always verify, env var has NO effect here.
    if not _validate_confirmation_token(
        token,
        scope=scope,
        target_url=target_url,
        app=app,
        row_counts=row_counts,
    ):
        from dbsprout.web.errors import (  # noqa: PLC0415
            web_error_write_guard_rejected,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=web_error_write_guard_rejected().to_dict(),
        )


class PreviewRequest(BaseModel):
    """Request body for ``POST /api/insert/preview``.

    Mirrors :class:`InsertRequest` — only ``tables`` is meaningful here;
    forbidding extra keys keeps the contract tight (mirrors siblings).
    """

    model_config = ConfigDict(extra="forbid")

    tables: list[str] | None = Field(default=None)


@insert_router.post("/api/insert/preview")
async def insert_preview_endpoint(request: Request, body: PreviewRequest) -> dict[str, Any]:
    """Issue a short-lived, single-use, scope-bound confirmation token.

    Guards:

    * No target on workspace → ``409 NO_CONNECTION``.
    * No generation result on workspace → ``409 NO_RUN``.
    * Unknown table in ``tables[]`` → ``422``.

    Response shape:

    .. code-block:: json

        {
          "target": "<redacted DSN>",
          "dialect": "postgresql",
          "scope": [{"table": "users", "row_count": 100}, ...],
          "total_rows": 250,
          "confirmation_token": "<base64url(payload).base64url(sig)>"
        }

    The token payload carries only hashes (``target_hash``,
    ``scope_hash``) plus a ``nonce`` and ``exp`` — never the raw DSN.
    """
    from dbsprout.web.errors import (  # noqa: PLC0415
        raise_web_error,
        web_error_no_connection,
        web_error_no_run,
    )
    from dbsprout.web.workspace import _redact_url  # noqa: PLC0415

    workspace = _workspace(request)
    raw_target = workspace.peek_target_url()
    if raw_target is None:
        raise_web_error(request, web_error_no_connection())
    result = workspace.get_last_result()
    if result is None or not result.tables_data:
        raise_web_error(request, web_error_no_run())
    assert result is not None  # narrowed
    known = set(result.tables_data.keys())
    if body.tables:
        unknown = [t for t in body.tables if t not in known]
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Unknown tables: {', '.join(sorted(unknown))}",
            )
    insertion_order_scope, _scope_warnings = _resolve_scope(result, body.tables)
    row_counts = {t: len(result.tables_data.get(t, [])) for t in insertion_order_scope}

    # Build + sign the token.
    assert raw_target is not None  # narrowed
    secret = _get_write_guard_secret(request.app)
    nonce = secrets.token_hex(16)
    payload: dict[str, Any] = {
        "target_hash": _hash_target(raw_target),
        "scope_hash": _hash_scope([(t, row_counts[t]) for t in insertion_order_scope]),
        "exp": int(time.time()) + _PREVIEW_TOKEN_TTL,
        "nonce": nonce,
    }
    token = _encode_token(payload, secret)
    issued = _get_issued_registry(request.app)
    _gc_issued_tokens(issued)  # lazy GC of expired nonces
    issued[nonce] = payload["exp"]

    return {
        "target": _redact_url(raw_target),
        "dialect": _detect_direct_dialect(raw_target),
        "scope": [{"table": t, "row_count": row_counts[t]} for t in insertion_order_scope],
        "total_rows": sum(row_counts.values()),
        "confirmation_token": token,
    }


# endregion write-guard


# ─────────────────────────────────────────────────────────────────────────
# region: POST /api/insert (S-136)
# ─────────────────────────────────────────────────────────────────────────


class InsertRequest(BaseModel):
    """Request body for ``POST /api/insert``.

    All three fields are optional. ``tables`` selects a subset (``None`` or
    empty list ⇒ insert all tables in FK-safe order); ``confirmation_token``
    is the S-137 HMAC scope-bound token; ``method`` (S-141) pins the writer
    strategy:

    * ``"auto"`` (default) — preserves the S-136 dispatch byte-for-byte
      (PG → ``PgCopyWriter`` if ``psycopg`` is installed else ``SaBatchWriter``;
      MySQL → ``MysqlLoadDataWriter`` if ``pymysql`` is installed else
      ``SaBatchWriter``; everything else → ``SaBatchWriter``).
    * ``"batch"`` — forces ``SaBatchWriter`` regardless of dialect (universal
      SQLAlchemy executemany — works on every supported DB).
    * ``"copy"`` — forces COPY / LOAD DATA on PostgreSQL / MySQL; any other
      dialect (sqlite / mssql / oracle / …) **or** PG/MySQL without the
      optional driver → ``409 METHOD_UNSUPPORTED`` (we never silently
      downgrade an explicit copy request — that would be a footgun).

    ``extra='forbid'`` rejects unexpected keys with ``422`` (mirrors
    ``GenerateRequest`` / ``ConnectRequest``).
    """

    model_config = ConfigDict(extra="forbid")

    tables: list[str] | None = Field(default=None)
    confirmation_token: str | None = Field(default=None)
    method: Literal["auto", "batch", "copy"] = Field(default="auto")


def _workspace(request: Request) -> Workspace:
    """Typed accessor for the shared session wired in ``app.py``."""
    return cast("Workspace", request.app.state.workspace)


def _detect_direct_dialect(url: str) -> str:
    """Detect database dialect from a connection URL prefix.

    Lifted verbatim from :mod:`dbsprout.cli.commands.generate` so the web
    layer routes to the same writer the CLI does — single source of truth
    for the dialect→writer policy.
    """
    lower = url.lower()
    if lower.startswith(("postgresql", "postgres")):
        return "postgresql"
    if lower.startswith("mysql"):
        return "mysql"
    if lower.startswith("sqlite"):
        return "sqlite"
    if lower.startswith("mssql"):
        return "mssql"
    return lower.split("://")[0].split("+")[0] if "://" in lower else "unknown"


def _select_writer(url: str) -> tuple[Any, str]:
    """Pick the writer for *url* — same auto-detect policy as the CLI.

    Returns ``(writer_instance, writer_class_name)``. The class name is
    surfaced in the API response so the user / Studio knows *which* writer
    will run before clicking Confirm.

    Policy (verbatim from ``cli/commands/generate.py::_run_direct_insert``):

    * ``postgresql`` → try ``import psycopg``; success ⇒
      :class:`~dbsprout.output.pg_copy.PgCopyWriter`, failure ⇒
      :class:`~dbsprout.output.sa_batch.SaBatchWriter` (fallback).
    * ``mysql`` → try ``import pymysql``; success ⇒
      :class:`~dbsprout.output.mysql_load_data.MysqlLoadDataWriter`, failure
      ⇒ :class:`~dbsprout.output.sa_batch.SaBatchWriter` (fallback).
    * ``sqlite`` / ``mssql`` / unknown ⇒
      :class:`~dbsprout.output.sa_batch.SaBatchWriter`.
    """
    from dbsprout.output.sa_batch import SaBatchWriter  # noqa: PLC0415

    dialect = _detect_direct_dialect(url)
    if dialect == "postgresql":
        try:
            import psycopg  # noqa: F401, PLC0415

            from dbsprout.output.pg_copy import PgCopyWriter  # noqa: PLC0415

            return PgCopyWriter(), "PgCopyWriter"
        except ImportError:
            return SaBatchWriter(), "SaBatchWriter"
    if dialect == "mysql":
        try:
            import pymysql  # type: ignore[import-untyped]  # noqa: F401, PLC0415

            from dbsprout.output.mysql_load_data import (  # noqa: PLC0415
                MysqlLoadDataWriter,
            )

            return MysqlLoadDataWriter(), "MysqlLoadDataWriter"
        except ImportError:
            return SaBatchWriter(), "SaBatchWriter"
    return SaBatchWriter(), "SaBatchWriter"


# ─────────────────────────────────────────────────────────────────────────
# region: insert method select (S-141)
# ─────────────────────────────────────────────────────────────────────────
# S-141 layers an explicit ``method ∈ {auto, batch, copy}`` choice on top
# of the S-136 auto-dispatch policy. The Studio modal exposes a small
# ``<select>`` so the user can pin the strategy (e.g. when COPY is blocked
# by an RDS policy → pick ``batch``).
#
# Design contract:
#
#   * ``method="auto"`` re-uses :func:`_select_writer` byte-for-byte. The
#     S-136 happy-path tests stay green without modification.
#   * ``method="batch"`` returns :class:`~dbsprout.output.sa_batch.SaBatchWriter`
#     for every dialect. SaBatch is universal — no guard fires.
#   * ``method="copy"`` returns the dialect-specific COPY / LOAD DATA writer
#     for postgresql / mysql ONLY when the optional driver is installed.
#     Any other dialect, or a missing optional driver, raises a typed
#     :class:`fastapi.HTTPException` with the
#     :class:`~dbsprout.web.errors.WebErrorCode.METHOD_UNSUPPORTED` envelope
#     (status 409; carries ``dialect`` / ``method`` / ``supported`` at the
#     top of ``detail``).
#
# The Wave 2 story (S-139 — multi-format export) also extends ``insert.py``
# in *its own* region (``# region: multi-format export (S-139)``); the two
# regions never collide because (a) S-139 adds a sibling endpoint, not a
# new dispatch branch, and (b) the conflict-avoidance rule keeps every
# edit inside its named region.
# ─────────────────────────────────────────────────────────────────────────


#: Supported writer strategies the Studio offers. Ordered: ``auto`` is the
#: default, ``batch`` is the universal fallback, ``copy`` is the fast path.
_METHOD_AUTO = "auto"
_METHOD_BATCH = "batch"
_METHOD_COPY = "copy"

#: Dialects that support the COPY / LOAD DATA fast path (when their
#: respective optional drivers are installed). The list is intentionally
#: closed — adding a dialect needs both a writer module under
#: ``dbsprout/output/`` and an entry here.
_COPY_SUPPORTED_DIALECTS: frozenset[str] = frozenset({"postgresql", "mysql"})


def _supported_methods_for(dialect: str) -> list[str]:
    """Return the methods the *dialect* can serve.

    ``auto`` is always supported (it picks SaBatch as a universal
    fallback). ``batch`` is always supported (SaBatch is universal).
    ``copy`` is only supported on dialects listed in
    :data:`_COPY_SUPPORTED_DIALECTS`. The list is returned in stable order
    so the envelope shape doesn't depend on set iteration order.
    """
    supported = [_METHOD_AUTO, _METHOD_BATCH]
    if dialect in _COPY_SUPPORTED_DIALECTS:
        supported.append(_METHOD_COPY)
    return supported


def _raise_method_unsupported(
    *,
    dialect: str,
    method: str,
    hint: str | None = None,
) -> None:
    """Raise the typed :class:`HTTPException` for METHOD_UNSUPPORTED.

    Lazy-imports :mod:`dbsprout.web.errors` to keep the module's import
    surface unchanged for the S-136 happy path (the import only fires on
    the failure branch).
    """
    from dbsprout.web.errors import (  # noqa: PLC0415
        web_error_method_unsupported,
    )

    err = web_error_method_unsupported(
        dialect=dialect,
        method=method,
        supported=_supported_methods_for(dialect),
        hint=hint,
    )
    raise HTTPException(status_code=err.status_code, detail=err.to_dict())


def _resolve_writer(
    dialect: str,
    method: str,
    *,
    url: str,
) -> tuple[Any, str]:
    """Resolve ``(writer_instance, writer_class_name)`` from *dialect* + *method*.

    Branches:

    * ``method="auto"`` ⇒ delegate to :func:`_select_writer` (S-136 policy).
    * ``method="batch"`` ⇒ ``SaBatchWriter`` (universal — no dialect check).
    * ``method="copy"`` + ``dialect="postgresql"`` ⇒ try
      ``import psycopg``; success ⇒ ``PgCopyWriter``; ImportError ⇒
      :func:`_raise_method_unsupported` with a driver-install hint.
    * ``method="copy"`` + ``dialect="mysql"`` ⇒ try ``import pymysql``;
      success ⇒ ``MysqlLoadDataWriter``; ImportError ⇒
      :func:`_raise_method_unsupported` with a driver-install hint.
    * ``method="copy"`` + any other dialect ⇒
      :func:`_raise_method_unsupported` (wrong-dialect path).

    *url* is forwarded to :func:`_select_writer` only when ``method="auto"``;
    on the explicit branches the URL is unused (the dialect + driver
    presence fully determine the choice).
    """
    from dbsprout.output.sa_batch import SaBatchWriter  # noqa: PLC0415

    if method == _METHOD_AUTO:
        return _select_writer(url)
    if method == _METHOD_BATCH:
        return SaBatchWriter(), "SaBatchWriter"
    # Remaining branch: the method is "copy" (validated by InsertRequest).
    if dialect == "postgresql":
        try:
            import psycopg  # noqa: F401, PLC0415

            from dbsprout.output.pg_copy import PgCopyWriter  # noqa: PLC0415

            return PgCopyWriter(), "PgCopyWriter"
        except ImportError:
            _raise_method_unsupported(
                dialect=dialect,
                method=method,
                hint=(
                    "Install the PostgreSQL driver (pip install psycopg) to "
                    "use method='copy', or pick method='batch'."
                ),
            )
    if dialect == "mysql":
        try:
            import pymysql  # noqa: F401, PLC0415

            from dbsprout.output.mysql_load_data import (  # noqa: PLC0415
                MysqlLoadDataWriter,
            )

            return MysqlLoadDataWriter(), "MysqlLoadDataWriter"
        except ImportError:
            _raise_method_unsupported(
                dialect=dialect,
                method=method,
                hint=(
                    "Install the MySQL driver (pip install pymysql) to use "
                    "method='copy', or pick method='batch'."
                ),
            )
    # Wrong-dialect path — sqlite / mssql / oracle / unknown.
    _raise_method_unsupported(dialect=dialect, method=method)
    # _raise_method_unsupported always raises; the line below is unreachable
    # but keeps mypy happy on the return-type contract.
    raise AssertionError("unreachable")  # pragma: no cover


# endregion insert method select


def _resolve_scope(
    result: GenerateResult,
    tables: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Resolve the FK-safe insertion scope + emit any FK-prerequisite warnings.

    Returns ``(insertion_order_scope, scope_warnings)``.

    * ``tables=None`` or empty ⇒ scope is ``result.insertion_order`` (full).
    * ``tables=[…]`` ⇒ scope is ``result.insertion_order`` *filtered* to the
      requested set (preserves FK-safe ordering — never re-sorts).

    For each table in the scope whose FK parents are not also in the scope,
    a warning string is added. This matches the FR-029 *"validated or
    warned"* language — we do NOT block, since the user may already have
    parent rows in the target from a prior run; the writer / DB engine will
    raise the real FK violation if one occurs.
    """
    full_order = result.insertion_order
    if not tables:
        return list(full_order), []

    requested = set(tables)
    scope = [t for t in full_order if t in requested]
    warnings: list[str] = []
    # Best-effort FK-prerequisite hint. We use ``insertion_order`` (FK-safe)
    # to derive parents: any table appearing *before* a scoped table in
    # ``full_order`` that is referenced by the scoped table's row dicts
    # could be a parent. Without a re-introspected ``DatabaseSchema``
    # we can't know FK columns precisely from result alone — so we lean on
    # a simple heuristic: any non-scoped table earlier in ``full_order``
    # whose name appears as a column suffix in a scoped table's row keys
    # (e.g. ``user_id`` ⇒ ``users``) is a likely parent. The warning is a
    # hint, not an authoritative check; the writer enforces real FK
    # constraints at the DB layer.
    scoped_set = set(scope)
    for t in scope:
        rows = result.tables_data.get(t, [])
        if not rows:
            continue
        column_names = set(rows[0].keys())
        for earlier in full_order:
            if earlier == t or earlier in scoped_set:
                continue
            # A FK column typically looks like ``<parent>_id`` or
            # ``<parent_singular>_id``. Match either parent name or its
            # de-pluralised stem (drop trailing ``s``).
            stem = earlier.rstrip("s") or earlier
            if f"{stem}_id" in column_names or f"{earlier}_id" in column_names:
                warnings.append(
                    f"table {t!r} references {earlier!r} but {earlier!r} is not in the "
                    f"insert scope; ensure parents exist in the target."
                )
    return scope, warnings


def _scrub(message: str, raw_url: str | None) -> str:
    """Strip a workspace target URL + its password from *message*. Never raises.

    Mirrors :func:`dbsprout.web.routers.generate._scrub` verbatim (copied,
    not imported, so the two routers remain decoupled — the policy is small
    and stable).
    """
    if not raw_url:
        return message
    from dbsprout.web.workspace import _redact_url  # noqa: PLC0415

    out = message.replace(raw_url, _redact_url(raw_url))
    try:
        import sqlalchemy as sa  # noqa: PLC0415

        password = sa.engine.make_url(raw_url).password
    except Exception:  # never let credential scrubbing raise
        password = None
    if password:
        out = out.replace(password, "***")
    return out


def _build_job_fn(  # noqa: PLR0913
    workspace: Workspace,
    result: GenerateResult,
    schema: DatabaseSchema,
    insertion_order_scope: list[str],
    writer: Any,
    raw_target_url: str,
) -> Callable[[Callable[[ProgressEvent], None], CancelToken], object]:
    """Build the blocking ``fn(progress_callback, cancel_token)`` for the job.

    Iterates the scope one table at a time so the closure can emit a pair
    of :class:`~dbsprout.generate.progress.ProgressEvent`\\ s (``table_start``
    / ``table_done``) around each writer call — that's how live per-table
    progress reaches the S-109 WebSocket without changing the writer
    signatures. Each per-table call is functionally identical to a single
    full-scope call (same INSERTs, same per-call transaction); the only
    cost is one extra connect per table — fine for O(10) tables on
    localhost.

    Cooperative cancel checked at the top of each iteration: a cancelled
    token raises :class:`~dbsprout.generate.progress.GenerationCancelled`
    which the :class:`~dbsprout.web.jobs.JobManager` maps to
    :class:`~dbsprout.web.jobs.JobStatus.CANCELLED`.

    On any writer exception the closure scrubs the message + re-raises a
    fresh :class:`RuntimeError` so the raw ``user:password`` never reaches
    :class:`~dbsprout.web.jobs.JobRecord.error` (DBS-139 forward note).

    ``workspace`` is accepted (rather than re-derived from ``request``)
    because the job runs on a worker thread — there's no ``request`` there.
    It is unused today; reserved for future state-writes (e.g. recording
    the last-inserted scope on the workspace so the Studio UI can render a
    "since last insert" diff in a follow-up story).
    """
    del workspace  # unused today; kept on the signature for the future hook

    def fn(
        progress_callback: Callable[[ProgressEvent], None],
        cancel_token: CancelToken,
    ) -> object:
        from dbsprout.generate.progress import (  # noqa: PLC0415
            GenerationCancelled,
            ProgressEvent,
            _is_cancelled,
        )
        from dbsprout.output.models import InsertResult  # noqa: PLC0415

        total = len(insertion_order_scope)
        running_rows = 0
        tables_inserted = 0
        # Per-table loop — gives us live progress events on the existing
        # S-107/S-109 surface without modifying any writer signature.
        for i, table_name in enumerate(insertion_order_scope):
            if _is_cancelled(cancel_token):
                raise GenerationCancelled(tables_done=i, tables_total=total)
            rows = result.tables_data.get(table_name, [])
            progress_callback(
                ProgressEvent(
                    phase="table_start",
                    table=table_name,
                    tables_done=i,
                    tables_total=total,
                    rows_in_table=0,
                    total_rows=running_rows,
                )
            )
            try:
                writer.write(
                    {table_name: rows},
                    schema,
                    [table_name],
                    raw_target_url,
                )
            except Exception as exc:
                # Scrub credentials before re-raising; ``record.error`` =
                # ``str(exc)`` so a scrubbed message is the only way to keep
                # the raw ``user:password`` out of the JobRecord.
                scrubbed = _scrub(str(exc) or type(exc).__name__, raw_target_url)
                raise RuntimeError(scrubbed) from exc
            running_rows += len(rows)
            tables_inserted += 1
            progress_callback(
                ProgressEvent(
                    phase="table_done",
                    table=table_name,
                    tables_done=i + 1,
                    tables_total=total,
                    rows_in_table=len(rows),
                    total_rows=running_rows,
                )
            )
        # Return an :class:`InsertResult` so the manager records meaningful
        # telemetry on ``JobRecord.result``. We deliberately do NOT replace
        # ``workspace.last_result`` (the generated data is still useful for
        # an export / re-insert in a follow-up story).
        return InsertResult(
            tables_inserted=tables_inserted,
            total_rows=running_rows,
            duration_seconds=0.0,
        )

    return fn


@insert_router.post("/api/insert")
async def insert_endpoint(request: Request, body: InsertRequest) -> dict[str, Any]:
    """Submit an insertion run as a background job; return ``{job_id, scope, ...}``.

    Returns immediately (non-blocking — does not await the job to
    completion). Guards (all return the typed envelope via
    :mod:`dbsprout.web.errors`):

    * No ``confirmation_token`` ⇒ ``403 WRITE_GUARD_REQUIRED``.
    * No target on the workspace ⇒ ``409 NO_CONNECTION``.
    * No generation result on the workspace ⇒ ``409 NO_RUN``.
    * Unknown ``tables[]`` entry ⇒ ``422`` (parameter-side).
    * Second concurrent submit ⇒ ``409`` (single-active — S-108).
    """
    from dbsprout.web.errors import (  # noqa: PLC0415
        raise_web_error,
        web_error_no_connection,
        web_error_no_run,
    )

    workspace = _workspace(request)

    # 1. No-connection guard.
    raw_target = workspace.peek_target_url()
    if raw_target is None:
        raise_web_error(request, web_error_no_connection())

    # 2. No-run guard.
    result = workspace.get_last_result()
    if result is None or not result.tables_data:
        raise_web_error(request, web_error_no_run())

    # 3. Unknown-table guard. ``result`` is non-None here.
    assert result is not None  # narrowed by the guard above
    known: set[str] = set(result.tables_data.keys())
    if body.tables:
        unknown = [t for t in body.tables if t not in known]
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Unknown tables: {', '.join(sorted(unknown))}",
            )

    # 4. Resolve scope (FK-safe order preserved).
    insertion_order_scope, scope_warnings = _resolve_scope(result, body.tables)

    # 5. Write-guard gate (S-137). Must come AFTER the request shape is
    # validated + scope is known, so the announced scope is what the HMAC
    # verification binds against. ``raw_target`` is non-None at this point.
    assert raw_target is not None  # narrowed by the guard above
    _require_confirmation_token(
        body.confirmation_token,
        scope=insertion_order_scope,
        target_url=raw_target,
        app=request.app,
        row_counts={t: len(result.tables_data.get(t, [])) for t in insertion_order_scope},
    )

    # 6. Select writer. S-141 layers an explicit method-pin on top of the
    #    S-136 auto-dispatch policy; ``method="auto"`` re-uses the original
    #    ``_select_writer`` policy byte-for-byte.
    dialect = _detect_direct_dialect(raw_target)
    writer, writer_name = _resolve_writer(dialect, body.method, url=raw_target)

    # 7. Build the job closure + submit (single-active — JobManager raises
    # ``JobError`` on a second concurrent submit, mapped to 409 below).
    from dbsprout.web.jobs import JobError  # noqa: PLC0415

    schema = workspace.get_schema()
    if schema is None:
        # Defence in depth — if the workspace had a target + a last_result
        # but somehow no schema (only reachable via a manual workspace
        # mutation), surface a typed envelope rather than a 500.
        raise_web_error(request, web_error_no_run())
    assert schema is not None  # narrowed above
    fn = _build_job_fn(
        workspace,
        result,
        schema,
        insertion_order_scope,
        writer,
        raw_target,
    )
    manager = request.app.state.job_manager
    try:
        job_id = await manager.submit("insert", fn)
    except JobError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A job is already running: {exc}",
        ) from exc

    # 8. Echo the announced scope so the Studio UI can render a confirmation
    # / preview pane even before S-137 lands the formal /api/insert/preview.
    return {
        "job_id": job_id,
        "scope": [
            {"table": t, "row_count": len(result.tables_data.get(t, []))}
            for t in insertion_order_scope
        ],
        "total_rows": sum(len(result.tables_data.get(t, [])) for t in insertion_order_scope),
        "writer": writer_name,
        # S-141: echo the resolved method so the Studio can render a
        # "you picked X, server ran X" confirmation badge.
        "method": body.method,
        "scope_warnings": scope_warnings,
    }


# endregion POST /api/insert


# ─────────────────────────────────────────────────────────────────────────
# region: POST /api/update-column (S-139)
# ─────────────────────────────────────────────────────────────────────────
# S-139 layers a sibling endpoint on top of the S-137 write-guard scheme:
# instead of inserting whole tables, ``POST /api/update-column`` pushes the
# most recently regenerated column for one table to the live target via
# :func:`dbsprout.output.column_update.update_column` (S-138). The scope-hash
# is bound to ``(table, column, row_count)`` rather than
# ``[(table, row_count), ...]`` so an insert-token can never be replayed
# here and vice versa.
#
# The flow mirrors POST /api/insert step-for-step, minus the job manager —
# column updates are O(rows) UPDATEs that fit in a single request thread.
# When a future story needs to background a large update it can lift the
# call into the existing ``JobManager`` (S-108) without touching this
# region's contract.
# ─────────────────────────────────────────────────────────────────────────


def _hash_update_column_scope(*, table: str, column: str, row_count: int) -> str:
    """Stable SHA-256 hex digest of the update-column scope.

    The canonical form is a JSON object with stable key order so identifier
    values can carry punctuation (``"users:role"`` etc.) without colliding
    with neighbouring fields. The scope is a single triple, not a set of
    pairs like the insert scope. Tests in ``test_update_column.py``
    re-derive this hash to mint valid tokens.
    """
    canonical = json.dumps(
        {"table": table, "column": column, "row_count": row_count},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_update_column_token(  # noqa: PLR0911, PLR0913 — guards mirror _validate_confirmation_token
    token: str,
    *,
    target_url: str,
    table: str,
    column: str,
    row_count: int,
    app: FastAPI | None = None,
) -> bool:
    """Verify *token* against the announced update-column scope.

    Returns ``True`` only if every check passes (see
    :func:`_validate_confirmation_token` for the per-field semantics). The
    only differences from the insert validator are the scope-hash inputs
    (``(table, column, row_count)`` vs. ``[(t, n)…]``) and the strict
    ``kind == "update_column"`` marker so a valid insert-preview token
    cannot be replayed here.
    """
    if app is None or not token:
        return False
    secret = _get_write_guard_secret(app)
    payload = _decode_token(token, secret)
    if payload is None:
        return False
    exp = payload.get("exp")
    nonce = payload.get("nonce")
    claimed_target = payload.get("target_hash")
    claimed_scope = payload.get("scope_hash")
    kind = payload.get("kind")
    if kind != "update_column":
        return False
    if not isinstance(exp, int):
        return False
    if exp <= int(time.time()):
        return False
    if not isinstance(nonce, str):
        return False
    if not isinstance(claimed_target, str) or not isinstance(claimed_scope, str):
        return False
    if not hmac.compare_digest(claimed_target, _hash_target(target_url)):
        return False
    expected = _hash_update_column_scope(table=table, column=column, row_count=row_count)
    if not hmac.compare_digest(claimed_scope, expected):
        return False
    issued = _get_issued_registry(app)
    if nonce not in issued:
        return False
    issued.pop(nonce, None)
    return True


def _require_update_column_token(  # noqa: PLR0913 — sibling of _require_confirmation_token
    token: str | None,
    *,
    target_url: str,
    table: str,
    column: str,
    row_count: int,
    app: FastAPI | None = None,
) -> None:
    """Enforce the write-guard gate on ``POST /api/update-column``.

    Mirrors :func:`_require_confirmation_token`: missing AND env unset →
    403 ``WRITE_GUARD_REQUIRED``; missing AND env set → bypass
    (test-only escape hatch); present but invalid → 403
    ``WRITE_GUARD_REJECTED`` regardless of the env var.
    """
    if not token:
        if _write_guard_disabled():
            return
        from dbsprout.web.errors import (  # noqa: PLC0415
            web_error_write_guard_required,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=web_error_write_guard_required().to_dict(),
        )
    if not _validate_update_column_token(
        token,
        target_url=target_url,
        table=table,
        column=column,
        row_count=row_count,
        app=app,
    ):
        from dbsprout.web.errors import (  # noqa: PLC0415
            web_error_write_guard_rejected,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=web_error_write_guard_rejected().to_dict(),
        )


class UpdateColumnRequest(BaseModel):
    """Request body for ``POST /api/update-column``.

    ``table`` + ``column`` identify the cell-set; ``confirmation_token`` is
    the S-139 HMAC scope-bound token (re-uses the S-137 helpers but binds
    a different scope). ``extra='forbid'`` rejects unexpected keys with
    ``422`` (mirrors siblings).
    """

    model_config = ConfigDict(extra="forbid")

    table: str = Field(..., min_length=1)
    column: str = Field(..., min_length=1)
    confirmation_token: str | None = Field(default=None)


def _extract_pk_value(
    row: dict[str, Any],
    pk_cols: list[str],
) -> Any:
    """Build the PK value the S-138 writer expects.

    * Single-column PK → scalar (e.g. ``row["id"]``).
    * Composite PK → tuple in declared order.

    The writer raises ``ColumnUpdateError(pk_arity_mismatch)`` if the
    composite tuple does not match the table's PK arity — we forward the
    raise upstream rather than re-check here.
    """
    if len(pk_cols) == 1:
        return row.get(pk_cols[0])
    return tuple(row.get(name) for name in pk_cols)


def _map_column_update_error_to_http(
    exc: Exception,
    request: Request,
) -> None:
    """Map a :class:`ColumnUpdateError` to a typed envelope.

    Closed mapping:

    * ``no_primary_key`` / ``pk_column_update`` → 409 ``CONSTRAINT_VIOLATION``.
    * ``unknown_table`` / ``unknown_column`` → 404 ``NOT_FOUND``.
    * ``unsafe_identifier`` / ``invalid_connection`` / ``update_failed`` /
      ``pk_arity_mismatch`` → 500 ``INTERNAL`` (these are server-side
      invariants the caller cannot fix by retry).
    """
    from dbsprout.output.column_update import ColumnUpdateError  # noqa: PLC0415
    from dbsprout.web.errors import (  # noqa: PLC0415
        raise_web_error,
        web_error_constraint_violation,
        web_error_internal,
        web_error_not_found,
    )

    if not isinstance(exc, ColumnUpdateError):  # pragma: no cover — defensive
        raise_web_error(request, web_error_internal(), original=exc)
        return
    code = exc.code
    table = exc.table
    detail = exc.detail
    if code in {"no_primary_key", "pk_column_update"}:
        raise_web_error(
            request,
            web_error_constraint_violation(table=table, column=detail, reason=code),
            original=exc,
        )
    elif code in {"unknown_table", "unknown_column"}:
        column = detail if code == "unknown_column" else None
        raise_web_error(
            request,
            web_error_not_found(table=table, column=column),
            original=exc,
        )
    else:
        # ``unsafe_identifier``, ``invalid_connection``, ``update_failed``,
        # ``pk_arity_mismatch`` — all server-side invariants. Log + 500.
        raise_web_error(request, web_error_internal(), original=exc)


@insert_router.post("/api/update-column")
async def update_column_endpoint(
    request: Request,
    body: UpdateColumnRequest,
) -> dict[str, Any]:
    """Push the most recently regenerated column to the live target database.

    Guards (all return the typed envelope via :mod:`dbsprout.web.errors`):

    * No target on the workspace ⇒ ``409 NO_CONNECTION``.
    * No generation result on the workspace, or no rows for *table* ⇒
      ``409 NO_REGEN``.
    * No ``confirmation_token`` (and the test escape hatch is unset) ⇒
      ``403 WRITE_GUARD_REQUIRED``.
    * Token present but invalid ⇒ ``403 WRITE_GUARD_REJECTED``.
    * PK-less table ⇒ ``409 CONSTRAINT_VIOLATION`` (``reason='no_primary_key'``).
    * Updating a PK column ⇒ ``409 CONSTRAINT_VIOLATION`` (``reason='pk_column_update'``).
    * Unknown table / column ⇒ ``404 NOT_FOUND``.

    Returns ``{rows_updated, table, column}`` on success.
    """
    from dbsprout.output.column_update import (  # noqa: PLC0415
        ColumnUpdateError,
        update_column,
    )
    from dbsprout.web.errors import (  # noqa: PLC0415
        raise_web_error,
        web_error_no_connection,
        web_error_no_regen,
    )

    workspace = _workspace(request)

    # 1. No-connection guard.
    raw_target = workspace.peek_target_url()
    if raw_target is None:
        raise_web_error(request, web_error_no_connection())

    # 2. No-regen guard — workspace must have a generation result AND the
    #    requested table must have at least one row available (i.e. the
    #    user regenerated something for it).
    result = workspace.get_last_result()
    if result is None or not result.tables_data:
        raise_web_error(request, web_error_no_regen())
    assert result is not None  # narrowed
    rows = result.tables_data.get(body.table) or []
    if not rows:
        raise_web_error(request, web_error_no_regen())

    # 3. Write-guard gate — token bound to (target, table, column, row_count).
    assert raw_target is not None  # narrowed
    row_count = len(rows)
    _require_update_column_token(
        body.confirmation_token,
        target_url=raw_target,
        table=body.table,
        column=body.column,
        row_count=row_count,
        app=request.app,
    )

    # 4. Schema must be available so the writer can resolve PK columns.
    #    Defence in depth — the regenerate flow always seeds both.
    schema = workspace.get_schema()
    if schema is None:
        raise_web_error(request, web_error_no_regen())
    assert schema is not None  # narrowed

    table_schema = schema.get_table(body.table)
    if table_schema is None:
        from dbsprout.web.errors import web_error_not_found  # noqa: PLC0415

        raise_web_error(request, web_error_not_found(table=body.table))
    assert table_schema is not None  # narrowed

    # 5. Build (pk_value, new_value) iterator from the in-memory rows. The
    #    writer validates identifiers + PK arity internally; we only have
    #    to translate the row-dict shape into the pair shape it expects.
    pk_cols = list(table_schema.primary_key)
    pairs: list[tuple[Any, Any]] = [
        (_extract_pk_value(row, pk_cols), row.get(body.column)) for row in rows
    ]

    # 6. Open a lazy engine + call the writer. We dispose the engine on
    #    every call — the connection pool overhead is negligible against
    #    the per-row UPDATE cost, and the workspace target may change
    #    between calls (a future story will wire a persistent engine).
    import sqlalchemy as sa  # noqa: PLC0415

    engine = sa.create_engine(raw_target)
    try:
        try:
            result_update = update_column(
                connection=engine,
                schema=schema,
                table=body.table,
                column=body.column,
                rows=pairs,
            )
        except ColumnUpdateError as exc:
            _map_column_update_error_to_http(exc, request)
            raise  # pragma: no cover — _map_column_update_error_to_http always raises
    finally:
        engine.dispose()

    return {
        "rows_updated": result_update.rows_updated,
        "table": body.table,
        "column": body.column,
    }


# endregion POST /api/update-column
