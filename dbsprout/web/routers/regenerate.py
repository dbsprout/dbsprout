"""``POST /api/regenerate`` — re-roll one column or one whole table (S-131).

The Studio user clicks **Regenerate** on a grid row / column header; the
handler validates a tiny ``{table, column?, reroll?}`` body, reads the loaded
schema + last :class:`~dbsprout.generate.orchestrator.GenerateResult` from the
per-session :class:`~dbsprout.web.workspace.Workspace` (S-111, wired on
``app.state.workspace``), and dispatches to the appropriate
:mod:`dbsprout.generate.regenerate` entry point:

* missing ``column`` ⇒
  :func:`dbsprout.generate.regenerate.regenerate_table` (S-128 — re-rolls the
  non-key cells of one table while keeping PKs byte-identical and re-sampling
  FKs against the *current* parent rows).
* present ``column`` ⇒
  :func:`dbsprout.generate.regenerate.regenerate_column` (S-129 — re-rolls a
  single column on one table; the ``reroll`` integer is forwarded as the
  ``nonce`` knob).

Sync vs job (the ``SYNC_REGEN_THRESHOLD`` knob)
-----------------------------------------------
Small re-rolls run inline on the request thread so the Studio gets the new
rows back in a single round-trip and can paint the grid immediately. Large
re-rolls are submitted to the :class:`~dbsprout.web.jobs.JobManager` (S-108)
and the route returns ``{kind: "job", job_id, status}``; progress streams over
the existing ``/ws/jobs/{job_id}`` (S-109).

The threshold is exposed as a module-level constant
(:data:`SYNC_REGEN_THRESHOLD`, default ``100_000``) so tests can patch it
down and a future story can lift it to a config knob without touching the
route surface.

Guards (all return typed envelopes via :mod:`dbsprout.web.errors`)
------------------------------------------------------------------
* No schema loaded ⇒ ``409 NO_SCHEMA``.
* No generation result on the workspace (or the requested table has no rows)
  ⇒ ``409 NO_RUN`` — reuses the existing taxonomy from S-136.
* Unknown table / column ⇒ ``404 NOT_FOUND``.
* PK / FK-referenced column re-roll ⇒ ``409 CONSTRAINT_VIOLATION``.
* Missing ``table`` / negative ``reroll`` / extra fields ⇒ ``422``
  (auto from Pydantic).

Seed source
-----------
The regenerate entry points are seeded — the route prefers
:meth:`~dbsprout.web.workspace.Workspace.get_last_seed` (set by the generate
router on a successful run, S-127). When no seed has been recorded yet
(e.g. tests that seed ``last_result`` directly) the route falls back to a
deterministic workspace-scoped sentinel (``0``) — same call → same output, so
no entropy is leaked from the test harness.

Lazy-import contract
--------------------
``dbsprout serve`` lazy-imports the web layer; importing ``dbsprout.cli.app``
must never pull FastAPI or the generation pipeline. This module imports only
stdlib + FastAPI (already gated by the ``[web]`` extra) at module level. The
:mod:`dbsprout.generate.regenerate` core, the :class:`Workspace`, the
:class:`GenerateResult` and the error factories are imported lazily inside
the handler.

Module owns its own :class:`~fastapi.APIRouter` (``regenerate_router``),
registered by :func:`dbsprout.web.app.create_app` inside a delimited region.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Callable

    from dbsprout.generate.progress import CancelToken, ProgressEvent
    from dbsprout.web.workspace import Workspace


regenerate_router = APIRouter()


#: Inline-vs-background threshold. When the affected row count is at or below
#: this value the route runs the regen inline on the request thread and
#: returns the fresh rows in the response. Above this value the regen is
#: submitted as a background job and the route returns ``{kind: "job",
#: job_id, status}``. ``100_000`` is fast enough on a single laptop core
#: that the Studio still feels snappy on a sync click; bigger tables go to
#: the job manager so the UI can show per-table progress.
SYNC_REGEN_THRESHOLD: int = 100_000


#: Sentinel seed used when the workspace has no recorded seed yet (e.g. a
#: test seeds ``last_result`` directly without going through the generate
#: route). Deterministic — same call → same output — so the AC's "byte-
#: identical re-roll for the same seed+nonce" invariant still holds.
_FALLBACK_SEED: int = 0


class RegenerateRequest(BaseModel):
    """Request body for ``POST /api/regenerate``.

    ``table`` is required. ``column`` (optional) switches between whole-table
    (``None``) and single-column re-roll. ``reroll`` is forwarded as the
    ``nonce`` to :func:`dbsprout.generate.regenerate.regenerate_column` —
    bumping it produces a different draw while keeping the rest of the
    pipeline deterministic. ``extra='forbid'`` keeps the contract tight
    (mirrors ``GenerateRequest`` / ``InsertRequest``).
    """

    model_config = ConfigDict(extra="forbid")

    table: str = Field(..., min_length=1)
    column: str | None = Field(default=None, min_length=1)
    reroll: int = Field(default=0, ge=0)


def _workspace(request: Request) -> Workspace:
    """Typed accessor for the shared session wired in ``app.py``."""
    return cast("Workspace", request.app.state.workspace)


def _resolve_seed(workspace: Workspace) -> int:
    """Pick the seed for the regen call: workspace last-seed > fallback."""
    last = workspace.get_last_seed()
    if last is not None:
        return last
    return _FALLBACK_SEED


def _replace_table_rows(
    workspace: Workspace,
    table: str,
    new_rows: list[dict[str, Any]],
) -> None:
    """Replace one table's rows on ``workspace.last_result`` immutably.

    ``GenerateResult`` is a frozen dataclass — we cannot mutate it in place.
    The helper builds a fresh ``tables_data`` dict, recomputes ``total_rows``,
    and stores a new :class:`GenerateResult` on the workspace. All other
    fields (insertion order, table timings, duration) are preserved.
    """
    from dataclasses import replace  # noqa: PLC0415

    current = workspace.get_last_result()
    if current is None:  # pragma: no cover — defensive; guarded upstream
        return
    new_tables_data = {**current.tables_data, table: new_rows}
    new_total = sum(len(rows) for rows in new_tables_data.values())
    workspace.set_last_result(
        replace(current, tables_data=new_tables_data, total_rows=new_total),
    )


def _map_regenerate_error_to_http(exc: Exception, request: Request) -> None:
    """Map a :class:`~dbsprout.generate.regenerate.RegenerateError` to a typed envelope.

    The mapping is closed:

    * ``unknown_table`` / ``unknown_column`` → ``404 NOT_FOUND``.
    * ``primary_key`` / ``fk_referenced`` → ``409 CONSTRAINT_VIOLATION``.
    * ``no_rows`` → ``409 NO_RUN`` (reuses the existing S-136 taxonomy — the
      request points at a table the workspace has no rows for, so the user
      needs to re-run generate first).
    """
    from dbsprout.web.errors import (  # noqa: PLC0415
        raise_web_error,
        web_error_constraint_violation,
        web_error_no_run,
        web_error_not_found,
    )

    reason = getattr(exc, "reason", None)
    table = getattr(exc, "table", "")
    column = getattr(exc, "column", None)
    if reason in {"unknown_table", "unknown_column"}:
        raise_web_error(
            request,
            web_error_not_found(table=table, column=column),
            original=exc,
        )
    elif reason in {"primary_key", "fk_referenced"}:
        raise_web_error(
            request,
            web_error_constraint_violation(
                table=table,
                column=column,
                reason=str(reason),
            ),
            original=exc,
        )
    elif reason == "no_rows":
        raise_web_error(request, web_error_no_run(), original=exc)
    else:  # pragma: no cover - defensive fallback, no known reason hits here
        raise_web_error(request, web_error_no_run(), original=exc)


def _call_regenerate(
    *,
    body: RegenerateRequest,
    workspace: Workspace,
    request: Request,
) -> list[dict[str, Any]]:
    """Invoke the right regen entry point and translate domain errors.

    Centralised so the sync path and the job closure share a single
    error-mapping site. Lazy-imports :mod:`dbsprout.generate.regenerate` to
    preserve the lazy-import contract — the module is not loaded until the
    first regenerate request actually fires.
    """
    from dbsprout.generate.regenerate import (  # noqa: PLC0415
        RegenerateError,
        regenerate_column,
        regenerate_table,
    )

    schema = workspace.get_schema()
    result = workspace.get_last_result()
    # Both are guaranteed non-None by guards in the handler; assert for mypy.
    assert schema is not None
    assert result is not None

    state = result.tables_data
    seed = _resolve_seed(workspace)
    try:
        if body.column is None:
            return regenerate_table(schema, state, body.table, seed=seed)
        return regenerate_column(
            schema,
            state,
            body.table,
            body.column,
            seed=seed,
            nonce=body.reroll,
        )
    except RegenerateError as exc:
        _map_regenerate_error_to_http(exc, request)
        # ``_map_regenerate_error_to_http`` always raises — the bare ``raise``
        # below is for the type-checker / belt-and-braces.
        raise  # pragma: no cover


def _build_job_fn(
    body: RegenerateRequest,
    workspace: Workspace,
    request: Request,
) -> Callable[[Callable[[ProgressEvent], None], CancelToken], object]:
    """Build the blocking ``fn(progress_callback, cancel_token)`` for the job.

    Emits a single ``table_start`` / ``table_done`` progress pair around the
    regen call (the underlying regen entry points are pure and do not accept
    a progress callback yet; emitting bookend events is enough for the S-109
    WebSocket to drive a per-table progress bar in the Studio).

    Cooperative cancel checked at the top of the function: a cancelled token
    raises :class:`~dbsprout.generate.progress.GenerationCancelled` which the
    :class:`~dbsprout.web.jobs.JobManager` maps to
    :class:`~dbsprout.web.jobs.JobStatus.CANCELLED`.
    """

    def fn(
        progress_callback: Callable[[ProgressEvent], None],
        cancel_token: CancelToken,
    ) -> object:
        from dbsprout.generate.progress import (  # noqa: PLC0415
            GenerationCancelled,
            ProgressEvent,
            _is_cancelled,
        )

        if _is_cancelled(cancel_token):
            raise GenerationCancelled(tables_done=0, tables_total=1)

        rows = workspace.get_last_result().tables_data.get(body.table, [])  # type: ignore[union-attr]
        progress_callback(
            ProgressEvent(
                phase="table_start",
                table=body.table,
                tables_done=0,
                tables_total=1,
                rows_in_table=0,
                total_rows=0,
            ),
        )
        new_rows = _call_regenerate(body=body, workspace=workspace, request=request)
        _replace_table_rows(workspace, body.table, new_rows)
        progress_callback(
            ProgressEvent(
                phase="table_done",
                table=body.table,
                tables_done=1,
                tables_total=1,
                rows_in_table=len(new_rows),
                total_rows=len(new_rows),
            ),
        )
        # Echo back the table + row count so the JobRecord.result carries
        # something meaningful for ``GET /api/jobs/{job_id}`` callers.
        return {
            "table": body.table,
            "column": body.column,
            "rows_affected": len(new_rows),
            "previous_row_count": len(rows),
        }

    return fn


@regenerate_router.post("/api/regenerate")
async def regenerate_endpoint(request: Request, body: RegenerateRequest) -> dict[str, Any]:
    """Re-roll one column or one whole table.

    Returns one of two shapes:

    * **Sync** (``rows_affected <= SYNC_REGEN_THRESHOLD``):

      .. code-block:: json

          {
            "kind": "sync",
            "table": "users",
            "column": null,
            "rows_affected": 100,
            "rows": [ ... ]
          }

    * **Job** (above threshold):

      .. code-block:: json

          {
            "kind": "job",
            "job_id": "<hex>",
            "status": "running"
          }

    Guards (all typed envelopes via :mod:`dbsprout.web.errors`):

    * No schema ⇒ ``409 NO_SCHEMA``.
    * No generation result on the workspace ⇒ ``409 NO_RUN``.
    * Unknown table / column ⇒ ``404 NOT_FOUND``.
    * PK / FK-referenced regen ⇒ ``409 CONSTRAINT_VIOLATION``.
    * Second concurrent job submit ⇒ ``409`` (single-active model — S-108).
    """
    from dbsprout.web.errors import (  # noqa: PLC0415
        raise_web_error,
        web_error_no_run,
        web_error_no_schema,
    )

    workspace = _workspace(request)

    # 1. No-schema guard. ``regenerate_table`` / ``regenerate_column`` both
    #    take the schema directly — guard at the route boundary so the
    #    error envelope is friendly + typed.
    if workspace.get_schema() is None:
        raise_web_error(request, web_error_no_schema())

    # 2. No-run guard. The regen entry points need ``state[table]`` to be
    #    non-empty — if the workspace has no result at all, fail fast with
    #    NO_RUN so the user knows to run generate first.
    result = workspace.get_last_result()
    if result is None or not result.tables_data:
        raise_web_error(request, web_error_no_run())
    assert result is not None

    # 3. Decide sync vs job from the affected row count. The whole-table and
    #    single-column paths both touch one table at a time, so the rough
    #    estimate is ``len(state[table])``.
    affected_rows = len(result.tables_data.get(body.table, []))
    use_job = affected_rows > SYNC_REGEN_THRESHOLD

    if not use_job:
        # 4a. Sync path — call inline, persist the new rows, return them.
        new_rows = _call_regenerate(body=body, workspace=workspace, request=request)
        _replace_table_rows(workspace, body.table, new_rows)
        return {
            "kind": "sync",
            "table": body.table,
            "column": body.column,
            "rows_affected": len(new_rows),
            "rows": new_rows,
        }

    # 4b. Job path — submit + return id + status. ``JobError`` on a second
    #     concurrent submit is mapped to a friendly 409.
    from dbsprout.web.jobs import JobError  # noqa: PLC0415

    fn = _build_job_fn(body, workspace, request)
    manager = request.app.state.job_manager
    try:
        job_id = await manager.submit("regenerate", fn)
    except JobError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A job is already running: {exc}",
        ) from exc
    return {
        "kind": "job",
        "job_id": job_id,
        "status": manager.get(job_id).status.value,
    }
