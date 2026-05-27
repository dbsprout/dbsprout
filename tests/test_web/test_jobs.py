"""JobManager / JobRecord tests (S-108).

The in-process job runner off-loads the blocking generation pipeline to a
Starlette threadpool as a fire-and-forget background task: ``submit`` returns a
``job_id`` immediately (the architecture's ``submit(kind, fn) -> job_id`` with a
separate cancel endpoint, so the browser can poll + cancel while the job runs),
tracks a single active job, forwards S-107 ``ProgressEvent``s into the record,
and exposes a cooperative cancel.

The manager lives in the ``[web]`` extra (it imports ``starlette.concurrency``),
so the module guards with ``importorskip`` for a core-only install. Async tests
use the ``anyio`` pytest marker (backend pinned to ``asyncio`` by ``conftest``).
Coordination between the event-loop thread and the worker thread uses
``threading.Event`` (not sleeps) so the cancel / single-active paths are
deterministic. ``manager.wait(job_id)`` awaits the background task to a terminal
state — a test-friendly join the future WS handler does not need.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import pytest

pytest.importorskip("starlette", reason="starlette absent (pip install dbsprout[web])")

from dbsprout.generate.progress import (
    CancelToken,
    GenerationCancelled,
    ProgressEvent,
)
from dbsprout.web.jobs import (
    JobError,
    JobManager,
    JobRecord,
    JobStatus,
    _CancelToken,
)

# ── JobRecord defaults (sync) ─────────────────────────────────────────


def test_job_record_defaults() -> None:
    rec = JobRecord(id="abc", kind="generate", started_at=datetime.now(tz=timezone.utc))
    assert rec.status is JobStatus.RUNNING
    assert rec.finished_at is None
    assert rec.error is None
    assert rec.result is None
    assert rec.events == []
    assert rec.latest_event is None


# ── unknown-id lookups raise a typed error (sync) ─────────────────────


def test_get_unknown_id_raises() -> None:
    with pytest.raises(JobError):
        JobManager().get("nope")


def test_cancel_unknown_id_raises() -> None:
    with pytest.raises(JobError):
        JobManager().cancel("nope")


@pytest.mark.anyio
async def test_wait_unknown_id_is_noop() -> None:
    # ``wait`` on an id with no background task simply returns (no raise).
    await JobManager().wait("nope")


# ── cooperative cancel token satisfies the S-107 Protocol (sync) ──────


def test_cancel_token_satisfies_s107_protocol_and_flips() -> None:
    token = _CancelToken()
    assert isinstance(token, CancelToken)  # structural: has is_cancelled()
    assert token.is_cancelled() is False
    token.cancel()
    assert token.is_cancelled() is True


# ── submit returns immediately + off-loads to a worker thread (async) ─


@pytest.mark.anyio
async def test_submit_returns_id_and_runs_fn_off_event_loop() -> None:
    main_id = threading.get_ident()
    observed: dict[str, int] = {}

    def fn(_cb: object, _tok: object) -> str:
        observed["thread"] = threading.get_ident()
        return "RESULT"

    mgr = JobManager()
    job_id = await mgr.submit("generate", fn)

    assert job_id  # returned before completion
    await mgr.wait(job_id)  # join the background task

    assert observed["thread"] != main_id  # ran off the event-loop thread
    rec = mgr.get(job_id)
    assert rec.status is JobStatus.SUCCEEDED
    assert rec.result == "RESULT"
    assert rec.kind == "generate"
    assert rec.finished_at is not None


# ── submit forwards S-107 progress events into the record (async) ─────


@pytest.mark.anyio
async def test_submit_forwards_progress_events_to_record() -> None:
    def fn(cb: object, _tok: object) -> None:
        emit = cb  # the manager-supplied progress_callback
        emit(ProgressEvent(phase="table_start", table="t", tables_total=1))  # type: ignore[operator]
        emit(  # type: ignore[operator]
            ProgressEvent(
                phase="table_done",
                table="t",
                tables_done=1,
                tables_total=1,
                rows_in_table=3,
                total_rows=3,
            )
        )

    mgr = JobManager()
    job_id = await mgr.submit("generate", fn)
    await mgr.wait(job_id)

    rec = mgr.get(job_id)
    assert [e.phase for e in rec.events] == ["table_start", "table_done"]
    assert rec.latest_event is not None
    assert rec.latest_event.phase == "table_done"
    assert rec.latest_event.total_rows == 3


# ── a raising fn marks the record failed, does not crash submit (async)


@pytest.mark.anyio
async def test_submit_failure_marks_record_failed() -> None:
    def fn(_cb: object, _tok: object) -> None:
        raise RuntimeError("boom")

    mgr = JobManager()
    job_id = await mgr.submit("generate", fn)
    await mgr.wait(job_id)

    rec = mgr.get(job_id)
    assert rec.status is JobStatus.FAILED
    assert rec.error is not None
    assert "boom" in rec.error
    assert rec.finished_at is not None


# ── cancel arms the token; GenerationCancelled unwind → cancelled ─────


@pytest.mark.anyio
async def test_cancel_marks_record_cancelled() -> None:
    started = threading.Event()

    def fn(_cb: object, tok: CancelToken) -> None:
        started.set()  # signal the worker is in-flight
        deadline = time.monotonic() + 5  # bounded: a cancel regression fails fast
        while not tok.is_cancelled():
            if time.monotonic() > deadline:
                raise AssertionError("cancel token never armed within 5s")
            time.sleep(0.001)
        raise GenerationCancelled(tables_done=1, tables_total=2)

    mgr = JobManager()
    job_id = await mgr.submit("generate", fn)

    await _run_sync(started.wait)  # ensure the worker is running, then cancel
    mgr.cancel(job_id)
    await mgr.wait(job_id)

    rec = mgr.get(job_id)
    assert rec.status is JobStatus.CANCELLED
    assert rec.finished_at is not None


# ── single active job: a second submit is rejected (async) ────────────


@pytest.mark.anyio
async def test_single_active_job_rejects_second_submit() -> None:
    release = threading.Event()
    started = threading.Event()

    def blocking_fn(_cb: object, _tok: object) -> str:
        started.set()
        release.wait(timeout=5)
        return "first"

    def quick_fn(_cb: object, _tok: object) -> str:
        return "later"

    mgr = JobManager()
    first_id = await mgr.submit("generate", blocking_fn)
    await _run_sync(started.wait)  # first job is now running on the worker

    with pytest.raises(JobError):
        await mgr.submit("generate", quick_fn)  # rejected: already active

    release.set()  # let the first finish
    await mgr.wait(first_id)
    assert mgr.get(first_id).status is JobStatus.SUCCEEDED

    # After the active job reached a terminal state, a new submit is accepted.
    third_id = await mgr.submit("generate", quick_fn)
    await mgr.wait(third_id)
    assert mgr.get(third_id).status is JobStatus.SUCCEEDED


# ── lazy-import contract: jobs.py pulls no generation/CLI/fastapi ─────


def test_jobs_module_has_no_eager_heavy_imports() -> None:
    """Importing the manager must not pull the orchestrator/service/fastapi.

    ``jobs.py`` only needs stdlib + ``starlette.concurrency`` + the pure
    ``dbsprout.generate.progress`` module. Importing the heavy generation
    pipeline (or fastapi) here would regress the ``dbsprout serve`` lazy-import
    contract.
    """
    probe = (
        "import sys\n"
        "import dbsprout.web.jobs  # noqa: F401\n"
        "bad = [m for m in ("
        "    'dbsprout.generate.orchestrator',"
        "    'dbsprout.core.service',"
        "    'fastapi',"
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
        f"dbsprout.web.jobs eagerly imported heavy modules: {result.stdout.strip()}"
    )


# ── app.state wiring (mirrors S-111 workspace tests) ──────────────────


def test_create_app_wires_job_manager() -> None:
    pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")
    from dbsprout.web.app import create_app  # noqa: PLC0415

    assert isinstance(create_app().state.job_manager, JobManager)


def test_separate_apps_get_independent_job_managers() -> None:
    pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")
    from dbsprout.web.app import create_app  # noqa: PLC0415

    app_a = create_app()
    app_b = create_app()
    assert app_a.state.job_manager is not app_b.state.job_manager


# ── helper: run a blocking callable off the event loop (test-side) ────


async def _run_sync(func: object) -> object:
    import anyio  # noqa: PLC0415

    return await anyio.to_thread.run_sync(func)  # type: ignore[arg-type]
