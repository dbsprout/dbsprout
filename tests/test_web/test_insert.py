"""POST /api/insert background-job submit endpoint tests (S-136).

This is Wave 1 of the Output & Insertion sub-epic. The route accepts an
optional ``tables`` subset + an ``confirmation_token`` (the latter is the
forward-handoff seam for S-137, which lands the real HMAC verification in
Wave 2). Inserts go through the existing dialect-aware writers in
``dbsprout/output/`` (PG COPY, MySQL LOAD DATA, SaBatch fallback) — no new
writer code, no new dialect→writer mapping.

The web stack lives in the optional ``[web]`` extra, so the module guards
with ``pytest.importorskip("fastapi")`` before importing FastAPI symbols
(mirrors the sibling web tests).
"""

from __future__ import annotations

import sqlite3
import threading
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


# ── helpers (mirror tests/test_web/test_generate.py) ────────────────────


def _make_app(state_db: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=state_db)


def _temp_sqlite(tmp_path: Path) -> str:
    """Create a 2-table sqlite DB and return its ``sqlite:///`` URL."""
    db_path = tmp_path / "target.db"
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


def _connect_and_generate(app: FastAPI, tmp_path: Path) -> str:
    """Connect the workspace + populate it with a generate result.

    Calls ``POST /api/connect`` (which is fully synchronous — sets the target
    + loads the schema) via the test client, then runs the generation
    pipeline *synchronously* (not via the background job manager) and stows
    the result directly on the workspace. This avoids the
    TestClient-vs-background-asyncio.Task lifecycle mismatch (TestClient
    runs the ASGI app in its own thread so ``asyncio.Task``\\ s spawned by
    ``submit`` cannot be reliably joined from the calling test thread).

    The synchronous shortcut is fine for these tests — the *insert*
    endpoint is the unit under test; the *generate* endpoint already has
    its own dedicated tests (``tests/test_web/test_generate.py``).
    """
    from dbsprout.config.models import DBSproutConfig  # noqa: PLC0415
    from dbsprout.core.service import generate as svc_generate  # noqa: PLC0415

    target_url = _temp_sqlite(tmp_path)
    client = TestClient(app)
    r1 = client.post("/api/connect", json={"url": target_url})
    assert r1.status_code == 200, r1.text

    # Run the generation pipeline *synchronously* in the test thread and
    # stow the result on the workspace exactly the way the production
    # generate-job closure would on its worker thread.
    workspace = app.state.workspace
    schema = workspace.get_schema()
    assert schema is not None, "schema should have been loaded by /api/connect"
    config = DBSproutConfig()
    result = svc_generate(
        schema,
        config,
        seed=7,
        default_rows=config.generation.default_rows,
        engine="heuristic",
    )
    workspace.set_last_result(result)
    return target_url


def _bypass_write_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test-only escape hatch — the production code path always requires a token."""
    monkeypatch.setenv("DBSPROUT_DISABLE_WRITE_GUARD", "1")


# ── Step 1: router-registration smoke ───────────────────────────────────


def test_insert_router_is_importable() -> None:
    from fastapi import APIRouter  # noqa: PLC0415

    from dbsprout.web.routers.insert import insert_router  # noqa: PLC0415

    assert isinstance(insert_router, APIRouter)


def test_insert_route_is_mounted(tmp_path: Path) -> None:
    """Mounted in create_app under /api/insert (S-136 region)."""
    app = _make_app(tmp_path / "state.db")
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/api/insert" in paths


# ── Step 4: write-guard from day one (no token → 403) ───────────────────


def test_insert_without_confirmation_token_is_403(tmp_path: Path) -> None:
    """The write path is closed unless the request carries a token.

    S-137 (Wave 2) lands the real HMAC validation; this story locks the
    contract so callers (Studio JS, curl, future tests) cannot reach a live
    INSERT without going through the gate.
    """
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post("/api/insert", json={})
    assert resp.status_code == 403, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["code"] == "WRITE_GUARD_REQUIRED"
    assert "Traceback" not in resp.text


# ── Step 5: no-connection guard ─────────────────────────────────────────


def test_insert_without_connection_is_409(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No target wired on the workspace → friendly 409 (NO_CONNECTION)."""
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    # NB: NO /api/connect call — workspace.peek_target_url() is None.
    resp = TestClient(app).post("/api/insert", json={})
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "NO_CONNECTION"
    assert "Traceback" not in resp.text


# ── Step 6: no-run guard ────────────────────────────────────────────────


def test_insert_without_generate_result_is_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Target wired but no generate run → 409 (NO_RUN)."""
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    target_url = _temp_sqlite(tmp_path)
    TestClient(app).post("/api/connect", json={"url": target_url})
    # NB: NO /api/generate call.
    resp = TestClient(app).post("/api/insert", json={})
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "NO_RUN"


# ── Step 7: unknown-table guard ─────────────────────────────────────────


def test_insert_unknown_table_is_422(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post("/api/insert", json={"tables": ["nonexistent_table"]})
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert "nonexistent_table" in str(detail).lower()


def test_insert_extra_field_is_422(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post("/api/insert", json={"unexpected": 1})
    assert resp.status_code == 422


# ── Step 8: scope resolution + writer selection (happy path) ────────────


def test_insert_submits_job_and_returns_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    resp = TestClient(app).post("/api/insert", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["job_id"], str)
    assert body["job_id"]
    scope = body["scope"]
    assert isinstance(scope, list)
    table_names = [entry["table"] for entry in scope]
    # FK-safe order: users before posts.
    assert table_names == ["users", "posts"]
    for entry in scope:
        assert isinstance(entry["row_count"], int)
        assert entry["row_count"] > 0
    assert body["total_rows"] == sum(e["row_count"] for e in scope)
    assert body["writer"] == "SaBatchWriter"  # sqlite → SaBatchWriter
    assert body["scope_warnings"] == []  # full insertion order: no missing parents


# ── Step 9: job actually writes rows + emits per-table progress ─────────


@pytest.mark.anyio
async def test_insert_job_writes_rows_and_emits_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the route through async client so the background task shares the loop."""
    import httpx  # noqa: PLC0415
    from httpx import ASGITransport  # noqa: PLC0415

    from dbsprout.web.jobs import JobStatus  # noqa: PLC0415

    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    # Reuse the helper's connect+generate flow (synchronous), capturing the
    # target URL so we can verify rows landed.
    target_url = _connect_and_generate(app, tmp_path)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/insert", json={})
        assert resp.status_code == 200, resp.text
        job_id = resp.json()["job_id"]

    await app.state.job_manager.wait(job_id)
    record = app.state.job_manager.get(job_id)
    assert record.status is JobStatus.SUCCEEDED, record.error
    # Per-table progress events forwarded → S-109 streams these.
    phases = [event.phase for event in record.events]
    assert phases.count("table_start") >= 2  # users + posts
    assert phases.count("table_done") >= 2

    # Rows actually present in the target — proves the writer ran, not just the closure.
    db_path = target_url.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    try:
        users_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        posts_count = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    finally:
        conn.close()
    assert users_count > 0
    assert posts_count > 0


# ── Step 10: subset insertion ───────────────────────────────────────────


@pytest.mark.anyio
async def test_insert_subset_inserts_only_selected_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx  # noqa: PLC0415
    from httpx import ASGITransport  # noqa: PLC0415

    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    target_url = _connect_and_generate(app, tmp_path)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/insert", json={"tables": ["users"]})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert [e["table"] for e in body["scope"]] == ["users"]
        job_id = body["job_id"]

    await app.state.job_manager.wait(job_id)

    db_path = target_url.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    try:
        users_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        posts_count = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    finally:
        conn.close()
    assert users_count > 0
    assert posts_count == 0  # posts excluded from scope → not inserted


@pytest.mark.anyio
async def test_insert_subset_with_missing_parent_emits_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inserting 'posts' alone (which FK-refs 'users') surfaces a scope warning."""
    import httpx  # noqa: PLC0415
    from httpx import ASGITransport  # noqa: PLC0415

    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    # First insert users so the FK constraint holds for the warning test
    # (otherwise the subset insert would correctly fail at the DB layer).
    TestClient(app).post("/api/insert", json={"tables": ["users"]})
    # Need to await the prior job before submitting the next (single-active).
    active_id = app.state.job_manager._active_id
    if active_id is not None:
        await app.state.job_manager.wait(active_id)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/insert", json={"tables": ["posts"]})
        assert resp.status_code == 200, resp.text
        warnings = resp.json()["scope_warnings"]
        assert any("users" in w and "posts" in w for w in warnings)


# ── Step 11: single-active 409 ──────────────────────────────────────────


@pytest.mark.anyio
async def test_insert_second_submit_while_active_is_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second concurrent submit while the insert job is running → 409."""
    import asyncio  # noqa: PLC0415

    import httpx  # noqa: PLC0415
    from httpx import ASGITransport  # noqa: PLC0415

    from dbsprout.output.sa_batch import SaBatchWriter  # noqa: PLC0415

    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)

    release = threading.Event()
    started = threading.Event()
    original = SaBatchWriter.write

    def blocking_write(self: Any, *args: Any, **kwargs: Any) -> Any:
        started.set()
        release.wait(timeout=5)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SaBatchWriter, "write", blocking_write)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/insert", json={})
        assert resp.status_code == 200, resp.text
        # Wait until the worker thread really entered the writer (job is now active).
        await asyncio.to_thread(started.wait, 5)

        resp2 = await client.post("/api/insert", json={})
        assert resp2.status_code == 409, resp2.text
        detail = resp2.json()["detail"].lower()
        assert "running" in detail or "active" in detail

        # Let the first job complete cleanly.
        release.set()
        await app.state.job_manager.wait(resp.json()["job_id"])


# ── Step 12: credential redaction on writer failure ─────────────────────


@pytest.mark.anyio
async def test_insert_failure_redacts_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writer raises an exception that embeds the workspace's raw target →
    JobRecord.error must not carry the password in clear."""
    import httpx  # noqa: PLC0415
    from httpx import ASGITransport  # noqa: PLC0415

    from dbsprout.output.sa_batch import SaBatchWriter  # noqa: PLC0415
    from dbsprout.web.jobs import JobStatus  # noqa: PLC0415

    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)

    # Override the workspace's raw target with a credentialed PG URL — the
    # writer path is monkeypatched to fail before any connect attempt.
    raw_target = "postgresql://alice:s3cretpw@db.invalid:5432/app"
    app.state.workspace.set_target_url(raw_target)

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        msg = f"writer blew up reaching {raw_target}"
        raise RuntimeError(msg)

    monkeypatch.setattr(SaBatchWriter, "write", boom)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Force SaBatchWriter selection by leaving the target as PG-style but
        # the dispatch path will try psycopg → falls back to SaBatch on most
        # CI envs. To make the test deterministic, monkeypatch _select_writer
        # to always return SaBatchWriter.
        from dbsprout.web.routers import insert as insert_module  # noqa: PLC0415

        def force_sabatch(_url: str) -> tuple[Any, str]:
            return SaBatchWriter(), "SaBatchWriter"

        monkeypatch.setattr(insert_module, "_select_writer", force_sabatch)

        resp = await client.post("/api/insert", json={})
        assert resp.status_code == 200, resp.text
        job_id = resp.json()["job_id"]

    await app.state.job_manager.wait(job_id)
    record = app.state.job_manager.get(job_id)
    assert record.status is JobStatus.FAILED
    assert record.error is not None
    assert "s3cretpw" not in record.error
    assert "***" in record.error  # the URL was scrubbed → some mask visible


# ── Step 8 cont.: writer-dispatch dialect branches (coverage) ───────────


def test_select_writer_dispatch_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each dialect branch in _select_writer routes to the right writer.

    SaBatch is the universal fallback; PG and MySQL branches are exercised by
    simulating the optional driver being installed or missing.
    """
    from dbsprout.web.routers.insert import _select_writer  # noqa: PLC0415

    # sqlite → SaBatch
    _, name = _select_writer("sqlite:///x.db")
    assert name == "SaBatchWriter"
    # mssql → SaBatch
    _, name = _select_writer("mssql+pyodbc://x:y@h/db")
    assert name == "SaBatchWriter"
    # unknown scheme → SaBatch
    _, name = _select_writer("oracle://x:y@h/db")
    assert name == "SaBatchWriter"

    # postgresql with psycopg present → PgCopyWriter
    import builtins  # noqa: PLC0415

    real_import = builtins.__import__

    def fake_import_psycopg(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "psycopg":
            import types  # noqa: PLC0415

            return types.ModuleType("psycopg")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import_psycopg)
    _, name = _select_writer("postgresql://x:y@h/db")
    assert name == "PgCopyWriter"
    monkeypatch.setattr(builtins, "__import__", real_import)

    # postgresql with psycopg missing → SaBatch fallback
    def fake_import_no_psycopg(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "psycopg":
            raise ImportError("no psycopg")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import_no_psycopg)
    _, name = _select_writer("postgresql://x:y@h/db")
    assert name == "SaBatchWriter"
    monkeypatch.setattr(builtins, "__import__", real_import)

    # mysql with pymysql present → MysqlLoadDataWriter
    def fake_import_pymysql(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pymysql":
            import types  # noqa: PLC0415

            return types.ModuleType("pymysql")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import_pymysql)
    _, name = _select_writer("mysql://x:y@h/db")
    assert name == "MysqlLoadDataWriter"
    monkeypatch.setattr(builtins, "__import__", real_import)

    # mysql with pymysql missing → SaBatch fallback
    def fake_import_no_pymysql(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pymysql":
            raise ImportError("no pymysql")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import_no_pymysql)
    _, name = _select_writer("mysql://x:y@h/db")
    assert name == "SaBatchWriter"


# ── coverage: direct unit tests on private helpers ──────────────────────


def test_validate_confirmation_token_stub_returns_true_for_non_empty() -> None:
    """The S-136 stub returns ``bool(token)`` — S-137 will replace with HMAC."""
    from dbsprout.web.routers.insert import _validate_confirmation_token  # noqa: PLC0415

    assert _validate_confirmation_token("anything", scope=["t"], target_url="u") is True
    assert _validate_confirmation_token("", scope=[], target_url="") is False


def test_require_confirmation_token_failed_validation_raises_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``_validate_confirmation_token`` rejects a non-empty token →
    the gate still raises 403. Locks the S-137 plug-in seam."""
    from fastapi import HTTPException  # noqa: PLC0415

    from dbsprout.web.routers import insert as insert_module  # noqa: PLC0415

    monkeypatch.setattr(
        insert_module,
        "_validate_confirmation_token",
        lambda token, scope, target_url: False,  # noqa: ARG005
    )
    with pytest.raises(HTTPException) as exc_info:
        insert_module._require_confirmation_token(
            "bogus-token", scope=["t"], target_url="sqlite:///x"
        )
    assert exc_info.value.status_code == 403
    detail = exc_info.value.detail
    assert isinstance(detail, dict)
    assert detail["code"] == "WRITE_GUARD_REQUIRED"


def test_resolve_scope_empty_rows_table_is_skipped() -> None:
    """A table whose ``tables_data`` slice is empty is still in the scope but
    contributes no warning (the heuristic skips empty row-lists)."""
    from dbsprout.generate.orchestrator import GenerateResult  # noqa: PLC0415
    from dbsprout.web.routers.insert import _resolve_scope  # noqa: PLC0415

    result = GenerateResult(
        tables_data={"users": [], "posts": [{"id": 1, "user_id": 1}]},
        insertion_order=["users", "posts"],
    )
    scope, warnings = _resolve_scope(result, ["users"])
    assert scope == ["users"]
    # 'users' has no rows so no warnings; 'posts' isn't in scope so nothing to inspect.
    assert warnings == []


def test_scrub_returns_message_unchanged_when_url_is_none() -> None:
    """``_scrub(message, None)`` is a pass-through (early return)."""
    from dbsprout.web.routers.insert import _scrub  # noqa: PLC0415

    assert _scrub("some message", None) == "some message"
    assert _scrub("", None) == ""


def test_scrub_handles_malformed_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_scrub`` swallows a SQLAlchemy parse error and still returns the
    (URL-substituted) message — credentials never leak even on a malformed URL."""
    import sqlalchemy as sa  # noqa: PLC0415

    from dbsprout.web.routers import insert as insert_module  # noqa: PLC0415

    # Force the SQLAlchemy password lookup to raise, exercising the
    # except-branch of the scrubber.

    def boom_make_url(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("malformed")

    monkeypatch.setattr(sa.engine, "make_url", boom_make_url)
    # _redact_url is the path that runs first; it tolerates the same error.
    # The message should still go through (no raise).
    out = insert_module._scrub("hello", "::garbage::")
    assert isinstance(out, str)


@pytest.mark.anyio
async def test_insert_job_respects_cancel_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancel signal raised before any table is processed unwinds via
    ``GenerationCancelled`` → ``JobStatus.CANCELLED``."""
    import httpx  # noqa: PLC0415
    from httpx import ASGITransport  # noqa: PLC0415

    from dbsprout.output.sa_batch import SaBatchWriter  # noqa: PLC0415
    from dbsprout.web.jobs import JobStatus  # noqa: PLC0415

    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)

    # Pre-cancel: arm the token before the first iteration of the per-table
    # loop runs. We replace the writer with one that fails the test if
    # invoked — if cancel works, the writer is never reached.
    def must_not_run(*_args: Any, **_kwargs: Any) -> Any:
        msg = "writer should not run after pre-cancel"
        raise AssertionError(msg)

    monkeypatch.setattr(SaBatchWriter, "write", must_not_run)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/insert", json={})
        assert resp.status_code == 200, resp.text
        job_id = resp.json()["job_id"]
        # Arm the cancel token on the manager BEFORE the worker thread picks
        # up the iteration. The single-active model gives us a deterministic
        # active_id immediately after submit.
        app.state.job_manager.cancel(job_id)

    await app.state.job_manager.wait(job_id)
    record = app.state.job_manager.get(job_id)
    assert record.status is JobStatus.CANCELLED


def test_insert_with_no_schema_but_with_target_returns_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defence-in-depth path — workspace target set + last_result set but
    schema cleared manually → 409 NO_RUN (typed envelope, not a 500)."""
    _bypass_write_guard(monkeypatch)
    app = _make_app(tmp_path / "state.db")
    _connect_and_generate(app, tmp_path)
    # Forcibly clear the schema while keeping target + last_result.
    app.state.workspace.schema = None
    resp = TestClient(app).post("/api/insert", json={})
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "NO_RUN"
