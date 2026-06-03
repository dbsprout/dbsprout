"""``POST /api/generate`` — submit the generation pipeline as a background job (S-124).

The user starts a generation run from the dashboard; the handler validates a tiny
``{engine?, seed?}`` body, reads the loaded schema from the per-session
:class:`~dbsprout.web.workspace.Workspace` (S-111, wired on ``app.state.workspace``),
and submits a job to the :class:`~dbsprout.web.jobs.JobManager` (S-108, wired on
``app.state.job_manager``). The submitted closure runs
:func:`dbsprout.core.service.generate` (S-106) with the manager-supplied
``progress_callback`` + ``cancel_token`` and returns the ``GenerateResult``; the
route returns ``{"job_id": ..., "seed": ...}`` immediately (non-blocking) so the
client can display + later re-use the seed.

Seed surface (S-127)
--------------------
``seed`` is ``int | None``. ``None`` (or an omitted field) means *"server,
pick one for me"*: the handler materialises a fresh non-negative 63-bit int via
:func:`secrets.randbits` (CSPRNG; bandit-clean), stores it on
:class:`~dbsprout.web.jobs.JobRecord` for later retrieval, and returns it in the
JSON response. An explicit ``int`` is honoured verbatim — combined with the
deterministic pipeline this enables the AC's *"same seed → byte-identical
output"* property end-to-end.

``GET /api/jobs/{job_id}`` (S-127) surfaces ``JobRecord`` metadata (status,
engine, seed, timestamps, error) as a small JSON envelope so the Studio
console can show "Seed: <n>" + a Copy-seed affordance after a run completes
without scraping internal state.

Non-blocking + single-active
----------------------------
The route ``await``\\ s :meth:`JobManager.submit` (which starts a fire-and-forget
background task and returns the id at once) and returns straight away — it never
awaits the job to completion. A second submit while a job is active raises S-108's
:class:`~dbsprout.web.jobs.JobError`, surfaced here as a friendly ``409``.

Progress wiring for S-109
-------------------------
The manager hands the ``fn`` closure a ``progress_callback`` and a ``cancel_token``.
The callback forwards each S-107 ``ProgressEvent`` into ``JobRecord.events`` (and
``latest_event``); the parallel sibling **S-109** streams those over a WebSocket.
This story only *wires the callback through* — it does not build the WS.

Credential redaction (FR-009 / DBS-139 forward note)
----------------------------------------------------
``service.generate`` runs over the already-introspected in-memory schema and opens
no DB connection, so a failure usually carries no URL. Defensively, the closure
wraps the pipeline call and scrubs the workspace's raw target (and its password)
out of any error message — reusing :func:`dbsprout.web.workspace._redact_url` — and
re-raises the scrubbed exception, so the raw ``user:password`` can never reach
``JobRecord.error`` or any API response.

This module owns its own :class:`~fastapi.APIRouter` (``generate_router``),
registered by ``create_app`` inside a delimited region. Heavy imports
(``core.service``, the config model, the redactor) are done lazily inside the
handler / closure to preserve the ``dbsprout serve`` lazy-import contract.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

#: Width of the seed materialised when the client says ``null``/omits ``seed``.
#: Constrained to 63 bits so the value always fits in a signed 64-bit int —
#: matches ``GenerationConfig.seed``'s ``>= 0`` constraint, lines up with the
#: AC's "non-negative 64-bit int" wording, and dodges the upper-bound corner of
#: ``2**64 - 1`` that some downstream sinks reject.
_SEED_BITS: int = 63

if TYPE_CHECKING:
    from collections.abc import Callable

    from dbsprout.generate.orchestrator import GenerateResult
    from dbsprout.generate.progress import CancelToken, ProgressEvent
    from dbsprout.web.workspace import Workspace

generate_router = APIRouter()

#: Engines the request may name (the four registered generation engines). An
#: unknown value is rejected at the boundary with a friendly 422.
_KNOWN_ENGINES: frozenset[str] = frozenset({"heuristic", "spec", "statistical", "finetuned"})


class GenerateRequest(BaseModel):
    """Request body for ``POST /api/generate``.

    Both fields are optional. ``engine`` defaults to ``"heuristic"``; ``seed``
    is ``int | None`` (S-127) — ``None`` / omitted means *"server picks a
    fresh non-negative 64-bit int via :func:`secrets.randbits` and returns it
    in the response"*. ``extra='forbid'`` rejects unexpected keys with ``422``
    (mirrors ``ConnectRequest``); when ``seed`` is provided it is constrained
    ``>= 0`` to match ``GenerationConfig.seed``; an unknown ``engine`` is
    rejected by the handler with a friendly ``422`` (kept as a handler check
    rather than an enum so the error message can list the supported engines).
    """

    model_config = ConfigDict(extra="forbid")

    engine: str = Field(default="heuristic")
    seed: int | None = Field(default=None, ge=0)

    @property
    def engine_is_known(self) -> bool:
        """True when :attr:`engine` is one of the registered generation engines."""
        return self.engine in _KNOWN_ENGINES


def _workspace(request: Request) -> Workspace:
    """Typed accessor for the shared session wired in ``app.py``."""
    return cast("Workspace", request.app.state.workspace)


def _scrub(message: str, raw_url: str | None) -> str:
    """Strip a workspace target URL + its password from *message*. Never raises.

    Reuses :func:`dbsprout.web.workspace._redact_url` (the same SQLAlchemy
    ``hide_password=True`` approach used across the codebase) to mask the URL,
    then defensively replaces the bare password substring as well — so even an
    error string that embeds only the password (not the whole URL) is scrubbed.
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


def _build_job_fn(
    body: GenerateRequest,
    workspace: Workspace,
    seed: int,
) -> Callable[[Callable[[ProgressEvent], None], CancelToken], object]:
    """Build the blocking ``fn(progress_callback, cancel_token)`` for the job.

    Captures the loaded schema and a default config with the **materialised**
    seed and the request's engine. The seed is passed explicitly (rather than
    pulled from ``body.seed``) because the handler may have materialised it
    server-side when the client sent ``None`` — the closure must capture the
    value the manager / JobRecord will record so the run is reproducible.

    Runs :func:`dbsprout.core.service.generate` with the manager-supplied
    progress / cancel hooks (S-108 invokes ``fn`` positionally), scrubs DB
    credentials from any error before re-raising, and stores the
    ``GenerateResult`` on the workspace on success.

    The ``progress_callback`` forwards S-107 ``ProgressEvent``\\ s into
    ``JobRecord.events`` — the parallel sibling **S-109** streams those over a
    WebSocket; this closure only wires the callback through.
    """
    from dbsprout.config.models import DBSproutConfig  # noqa: PLC0415

    schema = workspace.get_schema()
    config = DBSproutConfig()
    engine = body.engine
    default_rows = config.generation.default_rows
    raw_target = workspace.peek_target_url()

    def fn(
        progress_callback: Callable[[ProgressEvent], None],
        cancel_token: CancelToken,
    ) -> object:
        from dbsprout.core.service import generate  # noqa: PLC0415

        try:
            result = generate(
                schema,  # type: ignore[arg-type]  # guarded non-None by the handler
                config,
                seed=seed,
                default_rows=default_rows,
                engine=engine,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
            )
        except Exception as exc:  # scrub creds, then re-raise (manager records str)
            raise RuntimeError(_scrub(str(exc) or type(exc).__name__, raw_target)) from exc
        workspace.set_last_result(result)
        # S-131: stash the materialised seed so the regenerate route can pin
        # the per-table RNG to the same state on later re-rolls.
        workspace.set_last_seed(seed)
        return result

    return fn


@generate_router.post("/api/generate")
async def generate_endpoint(request: Request, body: GenerateRequest) -> dict[str, Any]:
    """Submit a generation run as a background job; return ``{job_id, seed}``.

    Returns immediately (non-blocking — does not await the job to completion).
    ``400`` if no schema is loaded; ``422`` for an unknown engine / malformed
    body; ``409`` if a generation job is already running (single-active model).

    Seed materialisation (S-127)
    ----------------------------
    When ``body.seed is None`` (client said ``null`` or omitted the field) the
    handler picks a fresh non-negative 63-bit int via :func:`secrets.randbits`
    (CSPRNG; bandit-clean) and threads it onto the :class:`JobRecord` and into
    the closure so the run is reproducible from this seed. The materialised
    value is **always** echoed back in the response so the Studio console can
    display it + offer a Copy-seed affordance regardless of which path the
    client took.
    """
    if not body.engine_is_known:
        allowed = ", ".join(sorted(_KNOWN_ENGINES))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown engine {body.engine!r}. Supported engines: {allowed}.",
        )

    workspace = _workspace(request)
    if workspace.get_schema() is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No schema loaded; connect to a database or upload a schema first.",
        )

    # Materialise the seed *before* building the job closure so the closure,
    # the JobRecord, and the response all reference the same value.
    seed: int = body.seed if body.seed is not None else secrets.randbits(_SEED_BITS)

    from dbsprout.web.jobs import JobError  # noqa: PLC0415

    fn = _build_job_fn(body, workspace, seed)
    manager = request.app.state.job_manager
    try:
        # S-110: thread engine + seed through to the JobRecord so the
        # state-write hook can map them onto the persisted RunRecord
        # without poking at the closure's captured locals. S-127: the
        # materialised seed (not body.seed, which may have been None) is
        # what we record.
        job_id = await manager.submit("generate", fn, engine=body.engine, seed=seed)
    except JobError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A generation job is already running: {exc}",
        ) from exc
    return {"job_id": job_id, "seed": seed}


# ── GET /api/jobs/{job_id} (S-127) ─────────────────────────────────────


def _isoformat(value: Any) -> str | None:
    """Serialise a ``datetime`` as ISO-8601 (or ``None`` for ``None``).

    The :class:`~dbsprout.web.jobs.JobRecord` carries timezone-aware UTC
    datetimes already (``_now`` uses ``datetime.now(tz=utc)``), so
    ``isoformat`` produces the canonical ``...+00:00`` suffix the client +
    tests can parse with :meth:`datetime.fromisoformat`.
    """
    if value is None:
        return None
    return cast("str", value.isoformat())


@generate_router.get("/api/jobs/{job_id}")
async def get_job(request: Request, job_id: str) -> dict[str, Any]:
    """Return a small JSON envelope describing a job (S-127).

    Surfaces ``JobRecord`` metadata — ``id``, ``kind``, ``status``, ``engine``,
    ``seed``, ``started_at``, ``finished_at``, ``error`` — so the Studio
    console can display the seed used (+ "Copy seed") after a run completes
    without having to scrape internal state or wait on the S-109 WebSocket
    final frame.

    Returns ``404`` with a friendly string ``detail`` on an unknown id; the
    :class:`~dbsprout.web.jobs.JobError` thrown by ``JobManager.get`` is mapped
    to the HTTP shape (no traceback leaks).
    """
    from dbsprout.web.jobs import JobError  # noqa: PLC0415

    manager = request.app.state.job_manager
    try:
        record = manager.get(job_id)
    except JobError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown job {job_id!r}.",
        ) from exc

    return {
        "id": record.id,
        "kind": record.kind,
        "status": record.status.value,
        "engine": record.engine,
        "seed": record.seed,
        "started_at": _isoformat(record.started_at),
        "finished_at": _isoformat(record.finished_at),
        "error": record.error,
    }


# ── GET /api/jobs/{job_id}/result (P4-4 / DBS-204) ─────────────────────


@generate_router.get("/api/jobs/{job_id}/result")
async def get_job_result(request: Request, job_id: str) -> dict[str, Any]:
    """Return the *real* per-table generated counts + durations (P4-4).

    The plain ``GET /api/jobs/{job_id}`` envelope carries only metadata; this
    dedicated, terminal-only endpoint surfaces the ``GenerateResult`` captured
    on ``JobRecord.result`` so the Studio summary reflects actual generated rows
    and timings instead of the ``/api/spec`` approximation.

    Shape::

        {
            "job_id": str,
            "total_rows": int,
            "total_tables": int,
            "total_duration_ms": int,
            "tables": [{"table_name": str, "row_count": int, "duration_ms": int}, ...],
        }

    The per-table rows come from ``GenerateResult.table_timings`` — a tuple of
    ``(table_name, row_count, generation_ms)`` triples captured by the
    orchestrator — so the counts are the *generated* counts (cross-checking
    ``len(tables_data[name])``) and the durations are real per-table
    milliseconds. ``total_duration_ms`` rounds ``duration_seconds`` to whole
    milliseconds.

    Errors:

    * ``404`` (friendly string ``detail``) on an unknown id — the
      :class:`~dbsprout.web.jobs.JobError` from ``JobManager.get`` is mapped to
      the HTTP shape (no traceback leak).
    * ``409`` when the job has not produced a result yet (still running, or
      failed / cancelled) — the real result only exists on success, so there is
      nothing to summarise; the client should wait for / re-check the job
      status.

    The result is pure generated data + counts (no DB target / DSN), so unlike
    the failure path of ``POST /api/generate`` there is no credential surface to
    scrub here. The orchestrator import stays lazy to preserve the
    ``dbsprout serve`` lazy-import contract.
    """
    from dbsprout.web.jobs import JobError  # noqa: PLC0415

    manager = request.app.state.job_manager
    try:
        record = manager.get(job_id)
    except JobError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown job {job_id!r}.",
        ) from exc

    if record.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Job {job_id!r} has no result yet (status {record.status.value!r}); "
                "the per-table summary is available only after a run succeeds."
            ),
        )

    result = cast("GenerateResult", record.result)
    tables = [
        {"table_name": name, "row_count": row_count, "duration_ms": generation_ms}
        for name, row_count, generation_ms in result.table_timings
    ]
    return {
        "job_id": record.id,
        "total_rows": result.total_rows,
        "total_tables": result.total_tables,
        "total_duration_ms": round(result.duration_seconds * 1000),
        "tables": tables,
    }
