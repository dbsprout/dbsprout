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

import os
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Request, status
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
# S-137 (Wave 2) lands the real HMAC verification + scope binding inside
# this region; the insert handler does not change. Until then the contract
# is:
#
#   * ``_require_confirmation_token`` raises ``403 WRITE_GUARD_REQUIRED``
#     when the token is missing, UNLESS the test-only escape hatch env var
#     ``DBSPROUT_DISABLE_WRITE_GUARD`` is set (production code never sets
#     this — it exists so S-136's own happy-path tests can drive the insert
#     without an HMAC-signed token).
#   * ``_validate_confirmation_token`` is a no-op stub returning ``True``
#     for any non-empty token. S-137 replaces the body with a real HMAC
#     verify + ``(target, scope)`` re-derivation; this signature is the
#     pre-agreed seam.
# ─────────────────────────────────────────────────────────────────────────

#: Test-only escape hatch env var. Production callers never set this.
_WRITE_GUARD_DISABLED_ENV = "DBSPROUT_DISABLE_WRITE_GUARD"


def _write_guard_disabled() -> bool:
    """``True`` when the test-only escape hatch env var is set to a truthy value.

    Kept as a private helper so the env-var name lives in exactly one place;
    S-137 will not touch this either.
    """
    raw = os.environ.get(_WRITE_GUARD_DISABLED_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _validate_confirmation_token(
    token: str,
    *,
    scope: list[str],
    target_url: str,
) -> bool:
    """Validate a confirmation token against the announced scope + target.

    **S-136 stub.** Returns ``True`` for any non-empty token — S-137 will
    replace the body with a real HMAC verify + a scope/target hash compare.
    The signature is the pre-agreed seam; callers (the insert handler) will
    not change when S-137 lands. ``scope`` + ``target_url`` are accepted
    here (even though the stub ignores them) so the S-137 implementation
    has the parameters it needs without a signature churn.
    """
    del scope, target_url  # used by S-137; recorded here to lock the contract.
    return bool(token)


def _require_confirmation_token(
    token: str | None,
    *,
    scope: list[str],
    target_url: str,
) -> None:
    """Enforce the write-guard gate (S-136 from day one).

    Raises :class:`fastapi.HTTPException` ``403 WRITE_GUARD_REQUIRED`` when
    the token is missing. When present, delegates to
    :func:`_validate_confirmation_token` (a stub today; HMAC verify under
    S-137). The test-only escape hatch ``DBSPROUT_DISABLE_WRITE_GUARD``
    short-circuits the missing-token path so S-136's happy-path tests can
    drive the insert without first synthesising an HMAC-signed token.
    Production callers never set this env var; the gate cannot be bypassed
    by a non-local actor.
    """
    if not token:  # None or empty string — both fail the gate.
        if _write_guard_disabled():
            return
        from dbsprout.web.errors import (  # noqa: PLC0415
            web_error_write_guard_required,
        )

        # The handler reroutes failed-validation paths via raise_web_error,
        # but for the missing-token path we raise a JSON ``HTTPException``
        # directly so the gate is uniform whether reached from the handler
        # or from a future direct caller (S-137 will use this same helper).
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=web_error_write_guard_required().to_dict(),
        )
    # Stub validation; S-137 will plug in real HMAC verification here.
    if not _validate_confirmation_token(token, scope=scope, target_url=target_url):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "WRITE_GUARD_REQUIRED",
                "message": "Confirmation token failed validation.",
            },
        )


# endregion write-guard


# ─────────────────────────────────────────────────────────────────────────
# region: POST /api/insert (S-136)
# ─────────────────────────────────────────────────────────────────────────


class InsertRequest(BaseModel):
    """Request body for ``POST /api/insert``.

    Both fields are optional. ``tables`` selects a subset (``None`` or
    empty list ⇒ insert all tables in FK-safe order); ``confirmation_token``
    is the S-137 forward-handoff seam (S-136 enforces *presence*; S-137
    lands the real HMAC validation). ``extra='forbid'`` rejects unexpected
    keys with ``422`` (mirrors ``GenerateRequest`` / ``ConnectRequest``).
    """

    model_config = ConfigDict(extra="forbid")

    tables: list[str] | None = Field(default=None)
    confirmation_token: str | None = Field(default=None)


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

    # 5. Write-guard gate (S-137 forward-handoff). Must come AFTER the
    # request shape is validated + scope is known, so the announced scope
    # is what S-137's real HMAC verification will bind against. ``raw_target``
    # is non-None at this point (guarded above).
    assert raw_target is not None  # narrowed by the guard above
    _require_confirmation_token(
        body.confirmation_token,
        scope=insertion_order_scope,
        target_url=raw_target,
    )

    # 6. Select writer (lifted CLI policy — no new writer code).
    writer, writer_name = _select_writer(raw_target)

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
        "scope_warnings": scope_warnings,
    }


# endregion POST /api/insert
