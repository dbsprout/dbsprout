"""ProgressHub + /ws/jobs/{job_id} WebSocket tests (S-109).

S-108's JobManager off-loads the blocking pipeline to a Starlette worker thread
and fires its progress_callback ON that worker thread. S-109's ProgressHub
bridges those worker-thread ProgressEvents to a WebSocket client over a per-job
``asyncio.Queue`` (worker enqueues via ``loop.call_soon_threadsafe``); the WS
route either replays a finished job's collected ``record.events`` or tails the
live queue, then sends a terminal frame and closes.

The hub/route live in the ``[web]`` extra; the module guards with
``importorskip``. Async hub tests use the ``anyio`` marker (backend pinned to
``asyncio`` by ``conftest``). The WS is exercised with FastAPI's
``TestClient.websocket_connect`` — note TestClient runs each request on a FRESH
event loop, so the integration test drives the job to completion first and
relies on the replay path (reading collected events, never a cross-loop queue).
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from dbsprout.generate.progress import ProgressEvent
from dbsprout.web.jobs import JobManager, JobRecord, JobStatus
from dbsprout.web.progress import ProgressHub, _tail_live, job_progress_ws, progress_ws_router

# ── hub unit: open/publish/close round-trip on the test's own loop ────


@pytest.mark.anyio
async def test_hub_open_publish_close_round_trip() -> None:
    hub = ProgressHub()
    hub.open("j1")
    queue = hub.queue("j1")
    assert queue is not None
    ev = ProgressEvent(phase="table_done", table="t", tables_done=1, tables_total=1)
    hub.publish("j1", ev)
    hub.close("j1")
    first = await queue.get()
    sentinel = await queue.get()
    assert first is ev
    assert sentinel is hub.SENTINEL


@pytest.mark.anyio
async def test_hub_publish_unknown_job_is_noop() -> None:
    # publishing before open (or for an unknown id) must not raise.
    ProgressHub().publish("nope", ProgressEvent(phase="table_start"))


@pytest.mark.anyio
async def test_hub_close_unknown_job_is_noop() -> None:
    ProgressHub().close("nope")


# ── direct coroutine tests for the live-tail + disconnect paths ───────
#
# TestClient.websocket_connect runs each request on a fresh loop, so a finished
# job always takes the *replay* branch there. To exercise the *live-tail* path
# (and the mid-stream disconnect), drive the handler coroutines directly on the
# test's own loop with a fake WebSocket + a real asyncio.Queue.


class _FakeWebSocket:
    """Minimal WebSocket double: records accept/close + sent JSON frames.

    ``send_json`` can be told to raise ``WebSocketDisconnect`` after *n* frames
    to model a client hanging up mid-stream.
    """

    def __init__(self, *, app: Any = None, disconnect_after: int | None = None) -> None:
        self.app = app
        self.sent: list[dict[str, Any]] = []
        self.accepted = False
        self.closed_code: int | None = -1  # -1 = "not closed yet"
        self._disconnect_after = disconnect_after

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, data: dict[str, Any]) -> None:
        if self._disconnect_after is not None and len(self.sent) >= self._disconnect_after:
            raise WebSocketDisconnect(code=1006)
        self.sent.append(data)

    async def close(self, code: int = 1000) -> None:
        self.closed_code = code


def _running_record(job_id: str = "live") -> JobRecord:
    return JobRecord(id=job_id, kind="generate", started_at=datetime.now(tz=timezone.utc))


@pytest.mark.anyio
async def test_tail_live_forwards_queued_events_then_terminal() -> None:
    record = _running_record()
    record.status = JobStatus.SUCCEEDED  # terminal status read for the final frame
    queue: asyncio.Queue[Any] = asyncio.Queue()
    sentinel = ProgressHub.SENTINEL
    for i in range(3):
        queue.put_nowait(ProgressEvent(phase="table_done", table=f"t{i}", tables_done=i + 1))
    queue.put_nowait(sentinel)

    ws = _FakeWebSocket()
    await _tail_live(ws, record, queue, sentinel)  # type: ignore[arg-type]

    phases = [f["phase"] for f in ws.sent]
    assert phases == ["table_done", "table_done", "table_done", "terminal"]
    assert [f.get("table") for f in ws.sent[:3]] == ["t0", "t1", "t2"]
    assert ws.sent[-1]["status"] == "succeeded"


@pytest.mark.anyio
async def test_tail_live_with_no_queue_falls_back_to_replay() -> None:
    record = _running_record()
    record.status = JobStatus.SUCCEEDED
    record.events.append(ProgressEvent(phase="table_done", table="only", tables_done=1))

    ws = _FakeWebSocket()
    await _tail_live(ws, record, None, ProgressHub.SENTINEL)  # type: ignore[arg-type]

    assert [f["phase"] for f in ws.sent] == ["table_done", "terminal"]
    assert ws.sent[0]["table"] == "only"


@pytest.mark.anyio
async def test_ws_handler_swallows_client_disconnect_midstream() -> None:
    # A running job → live-tail branch; the fake client disconnects after the
    # first frame. job_progress_ws must catch WebSocketDisconnect and return
    # without re-raising or trying to close again.
    hub = ProgressHub()
    manager = JobManager(progress_hub=hub)
    job_id = "live-job"
    record = _running_record(job_id)
    manager._records[job_id] = record
    hub.open(job_id)
    queue = hub.queue(job_id)
    assert queue is not None
    queue.put_nowait(ProgressEvent(phase="table_start", table="a"))
    queue.put_nowait(ProgressEvent(phase="table_done", table="a", tables_done=1))

    app = FastAPI()
    app.state.progress_hub = hub
    app.state.job_manager = manager
    ws = _FakeWebSocket(app=app, disconnect_after=1)

    await job_progress_ws(ws, job_id)  # type: ignore[arg-type]

    assert ws.accepted is True
    assert len(ws.sent) == 1  # only the first frame got through before disconnect
    assert ws.closed_code == -1  # returned early on disconnect; no close() call


# ── WS integration via TestClient (replay path) ──────────────────────


def _app_with_finished_job() -> tuple[TestClient, str]:
    """Build an app whose manager+hub ran a fake job to completion."""
    import anyio  # noqa: PLC0415

    hub = ProgressHub()
    manager = JobManager(progress_hub=hub)
    app = FastAPI()
    app.state.progress_hub = hub
    app.state.job_manager = manager
    app.include_router(progress_ws_router)

    def fake(cb: object, _tok: object) -> str:
        emit = cb  # manager-supplied progress_callback
        emit(ProgressEvent(phase="table_start", table="users", tables_total=2))  # type: ignore[operator]
        emit(  # type: ignore[operator]
            ProgressEvent(
                phase="table_done",
                table="users",
                tables_done=1,
                tables_total=2,
                rows_in_table=10,
                total_rows=10,
            )
        )
        return "RESULT"

    async def _drive() -> str:
        job_id = await manager.submit("generate", fake)
        await manager.wait(job_id)
        return job_id

    job_id = anyio.run(_drive)
    return TestClient(app), job_id


def test_ws_streams_collected_events_then_terminal_for_finished_job() -> None:
    client, job_id = _app_with_finished_job()
    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        frames = []
        while True:
            frame = ws.receive_json()
            frames.append(frame)
            if frame.get("phase") == "terminal":
                break
    phases = [f["phase"] for f in frames]
    assert phases == ["table_start", "table_done", "terminal"]
    assert frames[1]["table"] == "users"
    assert frames[1]["total_rows"] == 10
    assert frames[-1]["status"] == "succeeded"


def test_ws_terminal_frame_carries_failed_status_and_error() -> None:
    import anyio  # noqa: PLC0415

    hub = ProgressHub()
    manager = JobManager(progress_hub=hub)
    app = FastAPI()
    app.state.progress_hub = hub
    app.state.job_manager = manager
    app.include_router(progress_ws_router)

    def boom(_cb: object, _tok: object) -> None:
        raise RuntimeError("kaboom")

    async def _drive() -> str:
        job_id = await manager.submit("generate", boom)
        await manager.wait(job_id)
        return job_id

    job_id = anyio.run(_drive)
    client = TestClient(app)
    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        frame = ws.receive_json()
    assert frame["phase"] == "terminal"
    assert frame["status"] == "failed"
    assert "kaboom" in frame["error"]


def test_ws_unknown_job_closes_without_frames() -> None:
    hub = ProgressHub()
    app = FastAPI()
    app.state.progress_hub = hub
    app.state.job_manager = JobManager(progress_hub=hub)
    app.include_router(progress_ws_router)
    client = TestClient(app)

    with client.websocket_connect("/ws/jobs/does-not-exist") as ws:  # noqa: SIM117
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


# ── create_app wiring ─────────────────────────────────────────────────


def test_create_app_wires_progress_hub_shared_with_manager() -> None:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    app = create_app()
    assert isinstance(app.state.progress_hub, ProgressHub)
    # the wired manager must publish to the same hub instance
    assert app.state.job_manager._hub is app.state.progress_hub


def test_create_app_registers_ws_route() -> None:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    paths = {getattr(r, "path", None) for r in create_app().routes}
    assert "/ws/jobs/{job_id}" in paths


# ── lazy-import contract: progress.py pulls no generation/CLI code ─────


def test_progress_module_has_no_eager_generation_import() -> None:
    probe = (
        "import sys\n"
        "import dbsprout.web.progress  # noqa: F401\n"
        "bad = [m for m in ("
        "    'dbsprout.generate.orchestrator',"
        "    'dbsprout.core.service',"
        ") if m in sys.modules]\n"
        "print(bad)\n"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, trusted interpreter
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == "[]", (
        f"dbsprout.web.progress eagerly imported heavy modules: {result.stdout.strip()}"
    )
