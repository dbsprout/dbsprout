"""``POST /api/generate`` — submit the generation pipeline as a background job (S-124).

The user starts a generation run from the dashboard; the handler validates a tiny
``{engine?, seed?}`` body, reads the loaded schema from the per-session
:class:`~dbsprout.web.workspace.Workspace` (S-111, wired on ``app.state.workspace``),
and submits a job to the :class:`~dbsprout.web.jobs.JobManager` (S-108, wired on
``app.state.job_manager``). The submitted closure runs
:func:`dbsprout.core.service.generate` (S-106) with the manager-supplied
``progress_callback`` + ``cancel_token`` and returns the ``GenerateResult``; the
route returns ``{"job_id": ...}`` immediately (non-blocking).

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

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Callable

    from dbsprout.generate.progress import CancelToken, ProgressEvent
    from dbsprout.web.workspace import Workspace

generate_router = APIRouter()

#: Engines the request may name (the four registered generation engines). An
#: unknown value is rejected at the boundary with a friendly 422.
_KNOWN_ENGINES: frozenset[str] = frozenset({"heuristic", "spec", "statistical", "finetuned"})


class GenerateRequest(BaseModel):
    """Request body for ``POST /api/generate``.

    Both fields are optional with the AC's defaults (``engine="heuristic"``,
    ``seed=42``). ``extra='forbid'`` rejects unexpected keys with ``422``
    (mirrors ``ConnectRequest``); ``seed`` is constrained ``>= 0`` to match
    ``GenerationConfig.seed``; an unknown ``engine`` is rejected by the handler
    with a friendly ``422`` (kept as a handler check rather than an enum so the
    error message can list the supported engines).
    """

    model_config = ConfigDict(extra="forbid")

    engine: str = Field(default="heuristic")
    seed: int = Field(default=42, ge=0)

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
) -> Callable[[Callable[[ProgressEvent], None], CancelToken], object]:
    """Build the blocking ``fn(progress_callback, cancel_token)`` for the job.

    Captures the loaded schema and a default config (with the request's seed /
    engine). Runs :func:`dbsprout.core.service.generate` with the
    manager-supplied progress / cancel hooks (S-108 invokes ``fn`` positionally),
    scrubs DB credentials from any error before re-raising, and stores the
    ``GenerateResult`` on the workspace on success.

    The ``progress_callback`` forwards S-107 ``ProgressEvent``\\ s into
    ``JobRecord.events`` — the parallel sibling **S-109** streams those over a
    WebSocket; this closure only wires the callback through.
    """
    from dbsprout.config.models import DBSproutConfig  # noqa: PLC0415

    schema = workspace.get_schema()
    config = DBSproutConfig()
    seed = body.seed
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
        return result

    return fn


@generate_router.post("/api/generate")
async def generate_endpoint(request: Request, body: GenerateRequest) -> dict[str, Any]:
    """Submit a generation run as a background job; return ``{"job_id": ...}``.

    Returns immediately (non-blocking — does not await the job to completion).
    ``400`` if no schema is loaded; ``422`` for an unknown engine / malformed
    body; ``409`` if a generation job is already running (single-active model).
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

    from dbsprout.web.jobs import JobError  # noqa: PLC0415

    fn = _build_job_fn(body, workspace)
    manager = request.app.state.job_manager
    try:
        job_id = await manager.submit("generate", fn)
    except JobError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A generation job is already running: {exc}",
        ) from exc
    return {"job_id": job_id}
