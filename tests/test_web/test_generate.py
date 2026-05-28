"""POST /api/generate background-job submit endpoint tests (S-124).

The web stack lives in the optional ``[web]`` extra, so the module guards with
``pytest.importorskip("fastapi")`` before importing FastAPI symbols (mirrors the
sibling web tests). This is the first route that *starts work*: it composes the
S-106 core service, the S-108 job manager, and the S-111 workspace into a
non-blocking submit that returns a ``job_id`` immediately.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


def _make_app(state_db: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=state_db)


def _temp_sqlite(tmp_path: Path) -> str:
    """Create a 2-table sqlite DB and return its ``sqlite:///`` URL."""
    db_path = tmp_path / "gen.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute(
            "CREATE TABLE posts (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id))"
        )
        conn.commit()
    finally:
        conn.close()
    return f"sqlite:///{db_path}"


def _load_schema(app: FastAPI, tmp_path: Path) -> None:
    """Connect the app's workspace to a temp sqlite DB (reuses /api/connect)."""
    TestClient(app).post("/api/connect", json={"url": _temp_sqlite(tmp_path)})


# ── router registration / seam ─────────────────────────────────────────


def test_generate_router_is_importable() -> None:
    from fastapi import APIRouter  # noqa: PLC0415

    from dbsprout.web.routers.generate import generate_router  # noqa: PLC0415

    assert isinstance(generate_router, APIRouter)


# ── happy path: submit returns a job_id ────────────────────────────────


def test_generate_returns_job_id(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)
    resp = TestClient(app).post("/api/generate", json={"seed": 7})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["job_id"], str)
    assert body["job_id"]


def test_generate_defaults_when_body_empty(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)
    resp = TestClient(app).post("/api/generate", json={})
    assert resp.status_code == 200, resp.text
    assert resp.json()["job_id"]


# ── no-schema guard + input validation ─────────────────────────────────


def test_generate_without_schema_is_4xx(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")  # no schema loaded
    resp = TestClient(app).post("/api/generate", json={})
    assert 400 <= resp.status_code < 500, resp.text
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert "schema" in detail.lower()
    assert "Traceback" not in resp.text


def test_generate_unknown_engine_is_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)
    resp = TestClient(app).post("/api/generate", json={"engine": "bogus"})
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert "bogus" in detail.lower() or "engine" in detail.lower()


def test_generate_extra_field_is_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)
    resp = TestClient(app).post("/api/generate", json={"rows": 10})
    assert resp.status_code == 422


def test_generate_negative_seed_is_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)
    resp = TestClient(app).post("/api/generate", json={"seed": -1})
    assert resp.status_code == 422


# ── job actually runs; workspace.last_result is populated (async join) ──


@pytest.mark.anyio
async def test_generate_job_runs_and_sets_last_result(tmp_path: Path) -> None:
    """Drive the route through an async client so the fire-and-forget background
    task shares this event loop, then join it via the manager and assert the run
    succeeded, forwarded S-107 progress events, and stored the result."""
    import httpx  # noqa: PLC0415
    from httpx import ASGITransport  # noqa: PLC0415

    from dbsprout.web.jobs import JobStatus  # noqa: PLC0415

    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/generate", json={"seed": 7})
        assert resp.status_code == 200, resp.text
        job_id = resp.json()["job_id"]

    # The fire-and-forget task shares this event loop → join it deterministically.
    await app.state.job_manager.wait(job_id)

    record = app.state.job_manager.get(job_id)
    assert record.status is JobStatus.SUCCEEDED, record.error
    assert record.events  # S-107 progress events forwarded → S-109 streams these
    result = app.state.workspace.get_last_result()
    assert result is not None
    assert set(result.tables_data.keys()) == {"users", "posts"}


# ── credential redaction on job failure (DBS-139 forward note) ──────────


@pytest.mark.anyio
async def test_generate_failure_redacts_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pipeline error that embeds the workspace target must never leak the
    password into JobRecord.error (nor, by extension, any API response)."""
    import httpx  # noqa: PLC0415
    from httpx import ASGITransport  # noqa: PLC0415

    from dbsprout.core import service  # noqa: PLC0415
    from dbsprout.web.jobs import JobStatus  # noqa: PLC0415

    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)
    # Give the workspace a credentialed raw target (the closure scrubs against it).
    raw = "postgresql://alice:s3cretpw@db.invalid:5432/app"
    app.state.workspace.set_target_url(raw)

    def boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(f"pipeline blew up while reaching {raw}")

    monkeypatch.setattr(service, "generate", boom)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/generate", json={})
        assert resp.status_code == 200, resp.text  # submit still succeeds (async)
        job_id = resp.json()["job_id"]

    await app.state.job_manager.wait(job_id)
    record = app.state.job_manager.get(job_id)
    assert record.status is JobStatus.FAILED
    assert record.error is not None
    assert "s3cretpw" not in record.error  # password scrubbed
    assert "alice:***" in record.error or "***" in record.error  # redacted form present


# ── single active job: a second concurrent submit → 409 ─────────────────


@pytest.mark.anyio
async def test_generate_second_submit_while_active_is_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading  # noqa: PLC0415

    import anyio  # noqa: PLC0415
    import httpx  # noqa: PLC0415
    from httpx import ASGITransport  # noqa: PLC0415

    from dbsprout.core import service  # noqa: PLC0415

    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)

    release = threading.Event()
    started = threading.Event()
    real_generate = service.generate

    def blocking_generate(*args: object, **kwargs: object) -> object:
        started.set()
        release.wait(timeout=5)  # hold the single active job open
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(service, "generate", blocking_generate)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post("/api/generate", json={})
        assert first.status_code == 200, first.text
        first_id = first.json()["job_id"]

        await anyio.to_thread.run_sync(started.wait)  # ensure job #1 is in-flight

        second = await client.post("/api/generate", json={})
        assert second.status_code == 409, second.text  # single-active rejection
        assert "running" in second.json()["detail"].lower()

    release.set()
    await app.state.job_manager.wait(first_id)


# ── _scrub credential helper (direct unit tests) ───────────────────────


def test_scrub_no_target_returns_message_unchanged() -> None:
    """With no workspace target, the message passes through untouched."""
    from dbsprout.web.routers.generate import _scrub  # noqa: PLC0415

    assert _scrub("boom: something failed", None) == "boom: something failed"


def test_scrub_masks_url_and_password() -> None:
    from dbsprout.web.routers.generate import _scrub  # noqa: PLC0415

    raw = "postgresql://bob:hunter2@h:5432/db"
    out = _scrub(f"failed reaching {raw} (pw hunter2)", raw)
    assert "hunter2" not in out  # bare password also scrubbed
    assert "bob:***" in out


def test_scrub_url_without_password_is_left_intact() -> None:
    """A passwordless URL has nothing to mask; the message is returned unchanged."""
    from dbsprout.web.routers.generate import _scrub  # noqa: PLC0415

    raw = "sqlite:///tmp/x.db"
    msg = f"failed reaching {raw}"
    assert _scrub(msg, raw) == msg


def test_scrub_malformed_url_does_not_raise() -> None:
    """A target SQLAlchemy cannot parse falls back to a no-op password scrub
    (the redactor masks what it can) without raising."""
    from dbsprout.web.routers.generate import _scrub  # noqa: PLC0415

    raw = "not a url at all"
    out = _scrub("some error text", raw)
    assert isinstance(out, str)  # never raises


# ── router registration on the app ──────────────────────────────────────


def test_generate_route_registered_on_app(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/api/generate" in paths


# ── lazy-import contract: importing the router pulls no heavy modules ────


def test_generate_router_has_no_eager_heavy_imports() -> None:
    """Importing the router must not pull the generation pipeline / config model
    into the CLI startup path (heavy imports are lazy inside the handler/closure),
    preserving the ``dbsprout serve`` lazy-import contract (mirrors S-108's probe)."""
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    probe = (
        "import sys\n"
        "import dbsprout.web.routers.generate  # noqa: F401\n"
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
        f"generate router eagerly imported heavy modules: {result.stdout.strip()}"
    )


# ── S-110: completed runs persist via StateDB; job_history reads them ──


@pytest.mark.anyio
async def test_generate_job_persists_completed_run(tmp_path: Path) -> None:
    """Happy path: a finished /api/generate run lands in the wired state DB and
    is readable via app.state.get_state_db().get_runs() and JobManager.job_history()."""
    import httpx  # noqa: PLC0415
    from httpx import ASGITransport  # noqa: PLC0415

    from dbsprout.web.jobs import JobStatus  # noqa: PLC0415

    db_path = tmp_path / "state.db"
    app = _make_app(db_path)
    _load_schema(app, tmp_path)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/generate", json={"seed": 11, "engine": "heuristic"})
        assert resp.status_code == 200, resp.text
        job_id = resp.json()["job_id"]

    await app.state.job_manager.wait(job_id)
    record = app.state.job_manager.get(job_id)
    assert record.status is JobStatus.SUCCEEDED, record.error

    runs = app.state.get_state_db().get_runs()
    assert len(runs) == 1
    persisted = runs[0]
    assert persisted.engine == "heuristic"
    assert persisted.seed == 11
    assert persisted.total_rows >= 0
    assert persisted.total_tables == 2  # users, posts
    # table_stats present for each generated table
    stat_names = {s.table_name for s in persisted.table_stats}
    assert stat_names == {"users", "posts"}

    # The wired job_history() accessor reads the same DB.
    history = app.state.job_manager.job_history()
    assert len(history) == 1
    assert history[0].engine == "heuristic"


@pytest.mark.anyio
async def test_generate_failure_is_not_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed generation does NOT leave a run row in state.db."""
    import httpx  # noqa: PLC0415
    from httpx import ASGITransport  # noqa: PLC0415

    from dbsprout.core import service  # noqa: PLC0415
    from dbsprout.web.jobs import JobStatus  # noqa: PLC0415

    db_path = tmp_path / "state.db"
    app = _make_app(db_path)
    _load_schema(app, tmp_path)

    def boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("pipeline failed")

    monkeypatch.setattr(service, "generate", boom)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/generate", json={})
        assert resp.status_code == 200, resp.text
        job_id = resp.json()["job_id"]

    await app.state.job_manager.wait(job_id)
    record = app.state.job_manager.get(job_id)
    assert record.status is JobStatus.FAILED

    runs = app.state.get_state_db().get_runs()
    assert runs == []  # no fake-success row
    assert app.state.job_manager.job_history() == []
