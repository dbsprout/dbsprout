"""Live-progress WebSocket transport for the dashboard (S-109).

S-108's :class:`~dbsprout.web.jobs.JobManager` off-loads the blocking generation
pipeline to a Starlette worker thread and fires the job's ``progress_callback``
ON that worker thread. A WebSocket handler, by contrast, runs on the asyncio
event loop. :class:`ProgressHub` bridges the two: it owns a per-job
:class:`asyncio.Queue` and the event loop captured at submit time, and its
:meth:`publish` (called from the worker thread) hands events to the loop with
``loop.call_soon_threadsafe(queue.put_nowait, event)`` — the canonical
cross-thread handoff.

The WS endpoint ``GET /ws/jobs/{job_id}`` streams one JSON frame per
``ProgressEvent`` plus a final terminal frame, then closes. It is resilient to
*when* the client connects:

* **Finished job** — replays the job's already-collected ``record.events`` (read
  off the :class:`~dbsprout.web.jobs.JobRecord`, never a cross-loop queue), then
  the terminal frame. This also covers ``TestClient.websocket_connect``, which
  runs each request on a fresh event loop.
* **Running job** — tails the live queue (an unbounded ``asyncio.Queue`` buffers
  every event enqueued since submit, so a mid-job subscriber still drains the
  full backlog) until the terminal sentinel, then the terminal frame.

This module is imported only via ``dbsprout serve`` (lazy). It imports
FastAPI/Starlette (the ``[web]`` extra) and the *pure*
``dbsprout.generate.progress`` type; it does NOT import ``core/`` or the
generation orchestrator (a probe test asserts this) — the worker ``fn`` carries
the pipeline dependency, as in S-108.

NOTE: this is ``dbsprout/web/progress.py`` — the live progress **WebSocket**
(``/ws/jobs/{job_id}``), which the SPA consumes. The legacy S-092 SSE *page*
(``dbsprout/web/views/progress.py``) was removed in the P1c-5 cutover.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Final, cast

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

if TYPE_CHECKING:
    from dbsprout.generate.progress import ProgressEvent
    from dbsprout.web.jobs import JobManager, JobRecord

__all__ = ["ProgressHub", "progress_ws_router"]

#: Job statuses that mean the run is over (kept in sync with JobStatus values;
#: compared by string so this module need not import the JobStatus enum).
_TERMINAL_STATUSES: Final = frozenset({"succeeded", "failed", "cancelled"})


class ProgressHub:
    """Thread-safe bridge from worker-thread ``ProgressEvent``s to WS clients.

    One :class:`asyncio.Queue` per job id. :meth:`open` (called on the event-loop
    thread inside ``JobManager.submit``) captures the running loop and creates the
    queue. :meth:`publish` / :meth:`close` are called from the worker thread and
    marshal onto the loop via ``call_soon_threadsafe``. A late ``publish`` for an
    unknown id, or before :meth:`open`, is a harmless no-op (the job may have no
    hub, or already been reaped).
    """

    #: Queue sentinel signalling the stream is over (job reached a terminal state).
    SENTINEL: Final = object()

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Any]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def open(self, job_id: str) -> None:
        """Register a job's queue and bind the current running loop."""
        self._loop = asyncio.get_running_loop()
        self._queues[job_id] = asyncio.Queue()

    def queue(self, job_id: str) -> asyncio.Queue[Any] | None:
        """Return the job's live queue, or ``None`` if it was never opened."""
        return self._queues.get(job_id)

    def publish(self, job_id: str, event: ProgressEvent) -> None:
        """Enqueue *event* onto the job's queue from any thread (no-op if absent)."""
        self._schedule(job_id, event)

    def close(self, job_id: str) -> None:
        """Signal end-of-stream by enqueueing the sentinel (no-op if absent)."""
        self._schedule(job_id, self.SENTINEL)

    def _schedule(self, job_id: str, item: Any) -> None:
        queue = self._queues.get(job_id)
        if queue is None or self._loop is None:
            return
        self._loop.call_soon_threadsafe(queue.put_nowait, item)


progress_ws_router = APIRouter()


def _event_frame(event: ProgressEvent) -> dict[str, Any]:
    """Shape a ``ProgressEvent`` into a JSON-serialisable WS frame."""
    return event.model_dump(mode="json")


def _terminal_frame(record: JobRecord) -> dict[str, Any]:
    """Final frame carrying the job's terminal status (+ error if failed)."""
    return {
        "phase": "terminal",
        "status": record.status.value,
        "error": record.error,
    }


@progress_ws_router.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str) -> None:
    """Stream a job's progress events + a terminal frame, then close.

    Replays a finished job's collected events; otherwise tails the live queue.
    Closes with policy-violation code ``1008`` for an unknown ``job_id``.
    """
    await websocket.accept()
    manager = cast("JobManager", websocket.app.state.job_manager)
    hub = cast("ProgressHub", websocket.app.state.progress_hub)
    from dbsprout.web.jobs import JobError as _JobError  # noqa: PLC0415

    try:
        record = manager.get(job_id)
    except _JobError:
        await websocket.close(code=1008)
        return

    try:
        if record.status.value in _TERMINAL_STATUSES:
            await _replay_finished(websocket, record)
        else:
            await _tail_live(websocket, record, hub.queue(job_id), hub.SENTINEL)
    except WebSocketDisconnect:  # client hung up mid-stream
        return
    await websocket.close()


async def _replay_finished(websocket: WebSocket, record: JobRecord) -> None:
    """Send every collected event, then the terminal frame (finished job)."""
    for event in record.events:
        await websocket.send_json(_event_frame(event))
    await websocket.send_json(_terminal_frame(record))


async def _tail_live(
    websocket: WebSocket,
    record: JobRecord,
    queue: asyncio.Queue[Any] | None,
    sentinel: object,
) -> None:
    """Forward live queued events until the sentinel, then the terminal frame."""
    if queue is None:  # no live queue (no hub wired) → fall back to replay
        await _replay_finished(websocket, record)
        return
    while True:
        item = await queue.get()
        if item is sentinel:
            break
        await websocket.send_json(_event_frame(cast("ProgressEvent", item)))
    await websocket.send_json(_terminal_frame(record))
