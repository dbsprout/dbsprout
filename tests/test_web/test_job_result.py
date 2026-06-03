"""``GET /api/jobs/{id}/result`` — the richer job-result envelope (P4-4 / DBS-204).

The plain ``GET /api/jobs/{id}`` envelope (S-127) carries only metadata
(status / engine / seed / timestamps / error) — **not** the ``GenerateResult``.
This module covers the dedicated, terminal-only result endpoint that surfaces the
*real* per-table generated row counts + per-table / total duration from
``record.result`` (a :class:`~dbsprout.generate.orchestrator.GenerateResult`),
so the Studio summary reflects actual generated rows + timings rather than the
``/api/spec`` approximation.

Cases:

* **success** — after a real sqlite run, the endpoint returns per-table counts +
  durations and the totals, and the counts equal ``len(tables_data[name])``.
* **unknown id** — clean ``404`` with a string ``detail`` (no traceback leak).
* **not yet succeeded** — a running / failed / cancelled / result-less job yields a
  friendly ``409`` (the real result only exists on success).
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


# ── fixtures / helpers ─────────────────────────────────────────────────


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
    TestClient(app).post("/api/connect", json={"url": _temp_sqlite(tmp_path)})


# ── success: real per-table counts + durations + totals ────────────────


@pytest.mark.anyio
async def test_job_result_returns_real_counts_and_durations(tmp_path: Path) -> None:
    """After a real run, the endpoint surfaces actual generated counts + timings."""
    import httpx  # noqa: PLC0415
    from httpx import ASGITransport  # noqa: PLC0415

    from dbsprout.web.jobs import JobStatus  # noqa: PLC0415

    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        submit = await client.post("/api/generate", json={"seed": 7})
        assert submit.status_code == 200, submit.text
        job_id = submit.json()["job_id"]
        await app.state.job_manager.wait(job_id)
        record = app.state.job_manager.get(job_id)
        assert record.status is JobStatus.SUCCEEDED, record.error

        resp = await client.get(f"/api/jobs/{job_id}/result")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["job_id"] == job_id
    # Totals echo the GenerateResult.
    result = app.state.workspace.get_last_result()
    assert result is not None
    assert body["total_rows"] == result.total_rows
    assert body["total_tables"] == result.total_tables
    # Duration is exposed in milliseconds (rounded), non-negative.
    assert body["total_duration_ms"] == round(result.duration_seconds * 1000)
    assert body["total_duration_ms"] >= 0

    # One entry per generated table, real counts (= len of tables_data), with ms.
    tables = {t["table_name"]: t for t in body["tables"]}
    assert set(tables) == set(result.tables_data)
    for name, rows in result.tables_data.items():
        assert tables[name]["row_count"] == len(rows)
        assert tables[name]["duration_ms"] >= 0
    # The reported per-table counts sum to the reported total.
    assert sum(t["row_count"] for t in body["tables"]) == body["total_rows"]


# ── unknown id → 404 ───────────────────────────────────────────────────


def test_job_result_unknown_id_is_404(tmp_path: Path) -> None:
    """Unknown ids return a clean 404 with a string detail (no traceback)."""
    app = _make_app(tmp_path / "state.db")
    resp = TestClient(app).get("/api/jobs/does-not-exist/result")
    assert resp.status_code == 404
    assert "Traceback" not in resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert detail


# ── not-yet-succeeded → 409 ────────────────────────────────────────────


def test_job_result_without_result_is_409(tmp_path: Path) -> None:
    """A job with no result yet (no success) yields a friendly 409.

    Submitting a fabricated record straight onto the manager (status RUNNING,
    result None) models the "polled the result endpoint too early / on a
    failed run" case without racing a real pipeline.
    """
    from datetime import datetime, timezone  # noqa: PLC0415

    from dbsprout.web.jobs import JobRecord, JobStatus  # noqa: PLC0415

    app = _make_app(tmp_path / "state.db")
    manager = app.state.job_manager
    rec = JobRecord(
        id="fabricated",
        kind="generate",
        started_at=datetime.now(tz=timezone.utc),
        status=JobStatus.RUNNING,
    )
    manager._records[rec.id] = rec  # test reaches into the in-memory store

    resp = TestClient(app).get(f"/api/jobs/{rec.id}/result")
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert detail
    assert "Traceback" not in resp.text
