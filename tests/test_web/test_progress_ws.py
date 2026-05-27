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

import subprocess
import sys

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dbsprout.generate.progress import ProgressEvent
from dbsprout.web.jobs import JobManager
from dbsprout.web.progress import ProgressHub, progress_ws_router

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
    from starlette.websockets import WebSocketDisconnect  # noqa: PLC0415

    with client.websocket_connect("/ws/jobs/does-not-exist") as ws:  # noqa: SIM117
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


# ── create_app wiring ─────────────────────────────────────────────────


def test_create_app_wires_progress_hub_shared_with_manager() -> None:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    app = create_app()
    assert isinstance(app.state.progress_hub, ProgressHub)
    # the wired manager must publish to the same hub instance
    assert app.state.job_manager._hub is app.state.progress_hub  # type: ignore[attr-defined]


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
