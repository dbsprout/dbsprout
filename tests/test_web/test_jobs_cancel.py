"""POST /api/jobs/{job_id}/cancel route tests (S-126).

Cooperative cancel of the in-flight generate job. The route flips the S-107
``cancel_token`` exposed by S-108's :class:`~dbsprout.web.jobs.JobManager`; the
in-flight ``orchestrate()`` checks the token between per-table batches and
raises :class:`~dbsprout.generate.progress.GenerationCancelled`, which the
manager turns into a terminal ``CANCELLED`` :class:`~dbsprout.web.jobs.JobRecord`.

The web stack lives in the optional ``[web]`` extra, so the module guards with
``pytest.importorskip("fastapi")`` (mirroring sibling web tests). The route
returns ``{job_id, status}`` (status is ``cancelling`` or already-``cancelled``
depending on race with the worker; tests assert it is one of those values).
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.generate.progress import CancelToken, GenerationCancelled
from dbsprout.web.jobs import JobManager, JobRecord, JobStatus

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


def _make_app(state_db: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=state_db)


# ── router seam ───────────────────────────────────────────────────────


def test_jobs_router_is_importable() -> None:
    from fastapi import APIRouter  # noqa: PLC0415

    from dbsprout.web.routers.jobs import jobs_router  # noqa: PLC0415

    assert isinstance(jobs_router, APIRouter)


def test_jobs_route_registered_on_app(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/api/jobs/{job_id}/cancel" in paths


# ── happy path: cancel arms the token; worker yields ──────────────────


@pytest.mark.anyio
async def test_cancel_running_job_returns_200_and_status(
    tmp_path: Path,
) -> None:
    """Submit a blocking fn that watches the cancel_token; POST cancel; assert
    response carries ``{job_id, status}`` and the record reaches CANCELLED."""
    import anyio  # noqa: PLC0415
    import httpx  # noqa: PLC0415
    from httpx import ASGITransport  # noqa: PLC0415

    app = _make_app(tmp_path / "state.db")

    started = threading.Event()

    def blocking_fn(_cb: object, tok: CancelToken) -> None:
        started.set()
        deadline = time.monotonic() + 5
        while not tok.is_cancelled():
            if time.monotonic() > deadline:
                raise AssertionError("cancel token never armed")
            time.sleep(0.001)
        raise GenerationCancelled(tables_done=0, tables_total=1)

    manager = app.state.job_manager
    job_id = await manager.submit("generate", blocking_fn)

    # Ensure the worker is in-flight before we cancel.
    await anyio.to_thread.run_sync(started.wait)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(f"/api/jobs/{job_id}/cancel")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["job_id"] == job_id
        assert body["status"] in {"cancelling", "cancelled"}

    await manager.wait(job_id)
    assert manager.get(job_id).status is JobStatus.CANCELLED


# ── unknown id → 404 ──────────────────────────────────────────────────


def test_cancel_unknown_job_returns_404(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    resp = TestClient(app).post("/api/jobs/does-not-exist/cancel")
    assert resp.status_code == 404, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert "does-not-exist" in detail or "unknown" in detail.lower()


# ── already-terminal id → 409 ─────────────────────────────────────────


def _inject_record(app: FastAPI, status: JobStatus) -> str:
    """Insert a terminal JobRecord directly into the manager (white-box).

    Bypasses ``submit`` so we can land a record in an already-terminal status
    without racing a worker thread. The ``cancel`` route must reject these.
    """
    manager: JobManager = app.state.job_manager
    job_id = f"test-{status.value}"
    record = JobRecord(
        id=job_id,
        kind="generate",
        started_at=datetime.now(tz=timezone.utc),
        status=status,
        finished_at=datetime.now(tz=timezone.utc),
    )
    manager._records[job_id] = record
    return job_id


def test_cancel_succeeded_job_returns_409(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    job_id = _inject_record(app, JobStatus.SUCCEEDED)
    resp = TestClient(app).post(f"/api/jobs/{job_id}/cancel")
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert "succeeded" in detail.lower() or "terminal" in detail.lower()


def test_cancel_failed_job_returns_409(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    job_id = _inject_record(app, JobStatus.FAILED)
    resp = TestClient(app).post(f"/api/jobs/{job_id}/cancel")
    assert resp.status_code == 409, resp.text


def test_cancel_already_cancelled_job_returns_409(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    job_id = _inject_record(app, JobStatus.CANCELLED)
    resp = TestClient(app).post(f"/api/jobs/{job_id}/cancel")
    assert resp.status_code == 409, resp.text


# ── lazy-import contract (mirrors sibling tests) ──────────────────────


def test_jobs_router_has_no_eager_heavy_imports() -> None:
    """Importing the cancel router must not pull the generation pipeline / config
    model into the CLI startup path. Heavy modules belong inside handlers."""
    probe = (
        "import sys\n"
        "import dbsprout.web.routers.jobs  # noqa: F401\n"
        "bad = [m for m in ("
        "    'dbsprout.generate.orchestrator',"
        "    'dbsprout.core.service',"
        "    'dbsprout.config.models',"
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
        f"jobs router eagerly imported heavy modules: {result.stdout.strip()}"
    )
