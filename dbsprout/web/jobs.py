"""In-process background job runner for the web dashboard (S-108).

The dashboard's long-running work (generate, and later regenerate / insert)
must not block the asyncio event loop and must be cancellable while the browser
polls it. :class:`JobManager` runs **one** job at a time (single-user localhost
model): it off-loads the blocking pipeline to a Starlette worker thread
(``run_in_threadpool``), returns a ``job_id`` immediately, tracks the job's
lifecycle in an in-memory :class:`JobRecord`, forwards S-107 ``ProgressEvent``s
into that record, and exposes a cooperative cancel.

Scope (S-108)
-------------
Manager + record + wiring only. There is **no** WebSocket transport (S-109
streams ``record.events`` over a WS), **no** HTTP route (S-124 adds the generate
``POST`` that builds the ``fn`` closure from :func:`dbsprout.core.service.generate`),
and **no** persistence (writing finished runs to ``state.db`` is S-110). This
module stays in-memory and UI-agnostic.

Lazy-import contract
--------------------
``dbsprout serve`` lazy-imports the web layer; importing ``dbsprout.cli.app``
must never pull FastAPI or the generation pipeline. Accordingly this module
imports only stdlib, ``starlette.concurrency.run_in_threadpool``, and the *pure*
``dbsprout.generate.progress`` module (Pydantic + stdlib — it does NOT import the
orchestrator/engines). The blocking pipeline dependency rides in through the
``fn`` closure the caller (S-124) supplies; the manager treats ``fn``'s return
value as an opaque result reference.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from starlette.concurrency import run_in_threadpool

from dbsprout.generate.progress import GenerationCancelled

if TYPE_CHECKING:
    from collections.abc import Callable

    from dbsprout.generate.progress import ProgressEvent
    from dbsprout.state.models import RunRecord
    from dbsprout.web.progress import ProgressHub

__all__ = ["JobError", "JobManager", "JobRecord", "JobStatus"]

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Lifecycle states of a background job.

    ``QUEUED`` is part of the vocabulary (and S-109's streaming protocol) but in
    the single-active model :meth:`JobManager.submit` starts work immediately,
    so a record is created ``RUNNING`` and never actually waits in ``QUEUED``.
    ``SUCCEEDED`` / ``FAILED`` / ``CANCELLED`` are the terminal states.
    """

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED})


class JobError(ValueError):
    """Raised on an unknown ``job_id`` or when a second job is submitted.

    A typed error so the future routes can map it cleanly (single-active
    rejection → HTTP 409; unknown id → 404). Subclasses ``ValueError`` so
    generic ``except ValueError`` handlers still catch it.
    """


class _CancelToken:
    """A cooperative cancel signal satisfying S-107's :class:`CancelToken`.

    Backed by a :class:`threading.Event` so the flag is safely visible across
    the event-loop thread (which calls :meth:`cancel`) and the worker thread
    (which polls :meth:`is_cancelled` inside ``orchestrate``'s per-table loop).
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()


@dataclass
class JobRecord:
    """In-memory state of one background job (mutable, like ``Workspace``).

    Status transitions in place. The codebase's immutability rule targets
    *domain value objects* (``ProgressEvent`` / ``GenerateResult`` stay frozen);
    web-layer runtime/session state (``Workspace``, this record) is mutable by
    design. ``events`` collects every forwarded S-107 ``ProgressEvent`` (so S-109
    can stream them); ``latest_event`` is the convenience tail. ``result`` is the
    opaque ``fn`` return value — the manager is kind-agnostic, so it is typed
    ``object | None``; for the ``generate`` kind it is a
    :class:`~dbsprout.generate.orchestrator.GenerateResult` (the consumer casts).

    ``engine`` and ``seed`` (S-110) are optional metadata captured at
    :meth:`JobManager.submit` time so the state-write hook can populate
    :class:`~dbsprout.state.models.RunRecord` without having to peek into the
    submitter's closure. They default to ``None`` for back-compat with S-108
    callers that submit non-``generate`` kinds.
    """

    id: str
    kind: str
    started_at: datetime
    status: JobStatus = JobStatus.RUNNING
    finished_at: datetime | None = None
    error: str | None = None
    result: object | None = None
    events: list[ProgressEvent] = field(default_factory=list)
    latest_event: ProgressEvent | None = None
    engine: str | None = None
    seed: int | None = None


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class JobManager:
    """Runs a single background job at a time with lifecycle + cancellation.

    :meth:`submit` registers a ``RUNNING`` record, off-loads the blocking ``fn``
    to a worker thread as a fire-and-forget task, and returns the ``job_id``
    immediately so the caller can poll (:meth:`get`) and :meth:`cancel`.
    :meth:`wait` awaits a job's background task to its terminal state (used by
    tests and any caller that wants to join).
    """

    def __init__(
        self,
        progress_hub: ProgressHub | None = None,
        *,
        state_writer: Callable[[JobRecord], None] | None = None,
        state_reader: Callable[[], list[RunRecord]] | None = None,
    ) -> None:
        """Create a manager.

        *state_writer* (S-110) is invoked **once** with the finalized
        :class:`JobRecord` after a job reaches the ``SUCCEEDED`` terminal
        state. Failed / cancelled jobs are intentionally skipped — the state
        DB only records successful runs (no fake-success rows). A raise from
        the hook is swallowed (logged) so telemetry is strictly best-effort.

        *state_reader* (S-110) backs :meth:`job_history` — a thin accessor
        returning the read accessor's output verbatim. Both default to
        ``None`` so :class:`JobManager` keeps working without state plumbing
        (the S-108 / S-124 unit tests rely on this back-compat).
        """
        self._records: dict[str, JobRecord] = {}
        self._tokens: dict[str, _CancelToken] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._active_id: str | None = None
        self._hub = progress_hub
        self._state_writer = state_writer
        self._state_reader = state_reader

    # ── queries ────────────────────────────────────────────────────────
    def get(self, job_id: str) -> JobRecord:
        """Return the record for *job_id* or raise :class:`JobError`."""
        try:
            return self._records[job_id]
        except KeyError as exc:
            raise JobError(f"unknown job {job_id!r}") from exc

    def job_history(self) -> list[RunRecord]:
        """Return persisted completed runs (newest first), or ``[]``.

        Delegates to the injected ``state_reader``; if none is wired (e.g.
        bare ``JobManager()`` in unit tests) returns an empty list. The
        returned list is a fresh copy so callers cannot mutate the reader's
        cached state.
        """
        if self._state_reader is None:
            return []
        return list(self._state_reader())

    # ── lifecycle ──────────────────────────────────────────────────────
    async def submit(
        self,
        kind: str,
        fn: Callable[..., object],
        *,
        engine: str | None = None,
        seed: int | None = None,
    ) -> str:
        """Start *fn* as the single active background job; return its id.

        Rejects with :class:`JobError` if a job is already active
        (``queued``/``running``). The single-active check and the record
        registration run with no ``await`` between them, so under asyncio's
        single-threaded loop they are atomic (a concurrent ``submit`` only runs
        its check after this one has registered the active id).

        *engine* and *seed* (S-110) are optional metadata captured on the
        :class:`JobRecord` so the state-write hook can map them onto the
        persisted :class:`~dbsprout.state.models.RunRecord` without poking at
        the submitter's closure.
        """
        self._reject_if_active()
        job_id = uuid.uuid4().hex
        token = _CancelToken()
        record = JobRecord(
            id=job_id,
            kind=kind,
            started_at=_now(),
            engine=engine,
            seed=seed,
        )
        self._records[job_id] = record
        self._tokens[job_id] = token
        self._active_id = job_id
        if self._hub is not None:
            self._hub.open(job_id)
        self._tasks[job_id] = asyncio.create_task(self._run(record, token, fn))
        return job_id

    def cancel(self, job_id: str) -> None:
        """Arm the cooperative cancel token for *job_id*.

        Flips the token so the in-flight ``orchestrate()`` stops at its next
        per-table check and raises ``GenerationCancelled`` — :meth:`_run`'s
        handler then marks the record ``cancelled``. Cancelling an
        already-finished job is a harmless no-op on a dead token. Raises
        :class:`JobError` for an unknown id.
        """
        if job_id not in self._tokens:
            raise JobError(f"unknown job {job_id!r}")
        self._tokens[job_id].cancel()

    async def wait(self, job_id: str) -> None:
        """Await *job_id*'s background task to its terminal state."""
        task = self._tasks.get(job_id)
        if task is not None:
            await task

    # ── internals ──────────────────────────────────────────────────────
    def _reject_if_active(self) -> None:
        active = self._active_id
        if active is not None and self._records[active].status not in _TERMINAL:
            raise JobError("a job is already running")

    async def _run(
        self,
        record: JobRecord,
        token: _CancelToken,
        fn: Callable[..., object],
    ) -> None:
        """Off-load *fn* to a worker thread and finalize *record*."""

        def _on_progress(event: ProgressEvent) -> None:
            # Single writer thread (the worker) at a time; CPython list.append
            # and the attribute reassignment are GIL-atomic — no lock needed for
            # the single-active model. Also forward to the S-109 hub (if wired)
            # so a subscribed WebSocket sees the event live; publish marshals
            # onto the event loop via call_soon_threadsafe.
            record.events.append(event)
            record.latest_event = event
            if self._hub is not None:
                self._hub.publish(record.id, event)

        try:
            record.result = await run_in_threadpool(fn, _on_progress, token)
            record.status = JobStatus.SUCCEEDED
        except GenerationCancelled:
            record.status = JobStatus.CANCELLED
        except Exception as exc:  # reflect ANY pipeline failure in the record
            record.status = JobStatus.FAILED
            record.error = str(exc)
        finally:
            record.finished_at = _now()
            self._active_id = None
            # S-110: persist completed runs to the state DB. Only the
            # SUCCEEDED terminal state writes a row — FAILED / CANCELLED
            # would create a fake-success entry, which the AC explicitly
            # forbids. A raise from the hook is swallowed (best-effort
            # telemetry, matching the CLI ``_record_state`` contract); the
            # job's terminal status is therefore never regressed by a
            # state-write failure.
            if record.status is JobStatus.SUCCEEDED and self._state_writer is not None:
                try:
                    self._state_writer(record)
                except Exception as state_exc:
                    logger.warning(
                        "State-write hook failed for job %s (%s); "
                        "continuing — state telemetry is optional.",
                        record.id,
                        state_exc,
                    )
            # Terminal sentinel: lets a tailing WebSocket (S-109) stop and send
            # its final frame. Status is already set above, so the WS reads the
            # correct terminal status off the record.
            if self._hub is not None:
                self._hub.close(record.id)
