"""P5-9: a web ``/api/generate`` run persists quality telemetry to state.db.

S-110 already persists the *run* + ``table_stats`` (covered in ``test_generate``);
this module proves the P5-9 addition — the state-write hook now computes an
:class:`~dbsprout.quality.integrity.IntegrityReport` for the finished run (reusing
the same ``validate_integrity`` the ``/api/validate`` route runs) and records its
checks as ``quality_results``, so the SPA's Runs/Quality/Costs panels populate from
a web-only session. It also pins the best-effort contract: a state-write failure
never fails the generation job.

The web stack lives in the optional ``[web]`` extra, so the module guards with
``pytest.importorskip("fastapi")`` (mirrors the sibling web tests).
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

import httpx
from httpx import ASGITransport

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


def _make_app(state_db: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=state_db)


def _temp_sqlite(tmp_path: Path) -> str:
    """Create a 2-table sqlite DB (users ← posts) and return its URL."""
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
    from fastapi.testclient import TestClient  # noqa: PLC0415

    TestClient(app).post("/api/connect", json={"url": _temp_sqlite(tmp_path)})


async def _run_generate(app: FastAPI, **body: object) -> str:
    """POST /api/generate over an ASGI client and join the background job."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/generate", json=body)
        assert resp.status_code == 200, resp.text
        job_id: str = resp.json()["job_id"]
    await app.state.job_manager.wait(job_id)
    return job_id


# ── P5-9: a successful web run persists integrity quality_results ───────


@pytest.mark.anyio
async def test_web_generate_persists_quality_results(tmp_path: Path) -> None:
    """After a web generate succeeds, /api/quality reports integrity rows."""
    from dbsprout.web.jobs import JobStatus  # noqa: PLC0415

    db_path = tmp_path / "state.db"
    app = _make_app(db_path)
    _load_schema(app, tmp_path)

    job_id = await _run_generate(app, seed=11, engine="heuristic")
    assert app.state.job_manager.get(job_id).status is JobStatus.SUCCEEDED

    # The persisted run carries non-empty integrity quality_results.
    runs = app.state.get_state_db().get_runs()
    assert len(runs) == 1
    quality = runs[0].quality_results
    assert quality, "expected integrity quality_results to be persisted"
    assert all(qr.metric_type == "integrity" for qr in quality)

    # … and the read endpoints surface them for the SPA panels.
    from fastapi.testclient import TestClient  # noqa: PLC0415

    client = TestClient(app)

    runs_resp = client.get("/api/runs")
    assert runs_resp.status_code == 200
    assert runs_resp.json()["total_runs"] == 1

    quality_resp = client.get("/api/quality")
    assert quality_resp.status_code == 200
    qbody = quality_resp.json()
    assert qbody["found"] is True
    assert len(qbody["rows"]) >= 1
    assert all(row["metric_type"] == "integrity" for row in qbody["rows"])

    costs_resp = client.get("/api/costs")
    assert costs_resp.status_code == 200
    # Heuristic path makes no LLM calls — honest zero cost, no fabrication.
    assert costs_resp.json()["total_calls"] == 0


@pytest.mark.anyio
async def test_quality_rows_carry_pass_fail_status(tmp_path: Path) -> None:
    """Each persisted quality row exposes a pass/fail classification."""
    db_path = tmp_path / "state.db"
    app = _make_app(db_path)
    _load_schema(app, tmp_path)

    await _run_generate(app, seed=5, engine="heuristic")

    from fastapi.testclient import TestClient  # noqa: PLC0415

    rows = TestClient(app).get("/api/quality").json()["rows"]
    assert rows
    for row in rows:
        assert isinstance(row["passed"], bool)
        assert row["status"] in {"pass", "fail", "warn"}


# ── P5-9: best-effort — a state-write failure never fails the job ───────


@pytest.mark.anyio
async def test_generate_succeeds_when_state_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If persisting the run blows up, the generate job still SUCCEEDS (no 500)."""
    from dbsprout.state import writer  # noqa: PLC0415
    from dbsprout.web.jobs import JobStatus  # noqa: PLC0415

    db_path = tmp_path / "state.db"
    app = _make_app(db_path)
    _load_schema(app, tmp_path)

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("disk full")

    # Break the persist at the StateDB boundary inside record_job_run.
    monkeypatch.setattr(writer, "StateDB", _boom)

    with caplog.at_level(logging.WARNING):
        job_id = await _run_generate(app, seed=3)

    record = app.state.job_manager.get(job_id)
    assert record.status is JobStatus.SUCCEEDED, record.error
    # Nothing was persisted, but generation still produced a result.
    assert app.state.get_state_db().get_runs() == []
    assert app.state.workspace.get_last_result() is not None
    assert any("state" in r.message.lower() for r in caplog.records)


@pytest.mark.anyio
async def test_generate_succeeds_when_integrity_compute_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If computing the integrity report raises, the run still persists (no quality
    rows) and the job still SUCCEEDS — integrity is best-effort telemetry."""
    from dbsprout.quality import integrity  # noqa: PLC0415
    from dbsprout.web.jobs import JobStatus  # noqa: PLC0415

    db_path = tmp_path / "state.db"
    app = _make_app(db_path)
    _load_schema(app, tmp_path)

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("integrity exploded")

    monkeypatch.setattr(integrity, "validate_integrity", _boom)

    job_id = await _run_generate(app, seed=9)

    record = app.state.job_manager.get(job_id)
    assert record.status is JobStatus.SUCCEEDED, record.error
    # The run still persists (Runs panel works) — just without quality rows.
    runs = app.state.get_state_db().get_runs()
    assert len(runs) == 1
    assert runs[0].quality_results == []
