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

from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
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
