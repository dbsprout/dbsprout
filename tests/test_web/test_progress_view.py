"""Web SSE progress view tests (S-092).

The progress view (``dbsprout/web/views/progress.py``) ships in the optional
``[web]`` extra, so every test guards with ``pytest.importorskip("fastapi")``
*before* importing FastAPI symbols (mirrors ``tests/test_web/test_app.py``).

SSE safety: the endpoint's event generator is **bounded** — it polls a finite
number of times and always emits a terminal ``event: complete`` sentinel. Every
test here consumes a FINITE number of events (the body of a terminating stream
is fully buffered by ``TestClient.get``), so no test can hang on an unbounded
stream.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.state.db import StateDB
from dbsprout.state.models import RunRecord, TableStats

if TYPE_CHECKING:
    from pathlib import Path


def _make_client(state_db: Path) -> TestClient:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return TestClient(create_app(state_db_path=state_db))


def _parse_data_frames(body: str) -> list[dict]:
    """Extract and JSON-decode every ``data:`` frame from an SSE body."""
    payloads: list[dict] = []
    for line in body.splitlines():
        if line.startswith("data:"):
            chunk = line[len("data:") :].strip()
            if chunk:
                payloads.append(json.loads(chunk))
    return payloads


def _seed_completed_run(state_db: Path) -> None:
    db = StateDB(state_db)
    db.record_run(
        RunRecord(
            started_at=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
            completed_at=datetime(2026, 5, 20, 12, 0, 5, tzinfo=timezone.utc),
            duration_ms=5000,
            engine="heuristic",
            total_rows=4242,
            total_tables=2,
            seed=42,
            table_stats=[
                TableStats(
                    table_name="users", row_count=100, rows_per_sec=2000.0, generation_ms=50
                ),
                TableStats(
                    table_name="orders", row_count=4142, rows_per_sec=8284.0, generation_ms=500
                ),
            ],
        )
    )


# ── snapshot builder (pure function) ──────────────────────────────────


def test_build_snapshot_empty_is_idle() -> None:
    from dbsprout.web.views.progress import build_snapshot  # noqa: PLC0415

    snap = build_snapshot([])
    assert snap["status"] == "idle"
    assert snap["tables"] == []
    assert snap["overall_percent"] == 0


def test_build_snapshot_completed_run_is_100_percent() -> None:
    from dbsprout.web.views.progress import build_snapshot  # noqa: PLC0415

    run = RunRecord(
        started_at=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 5, 20, 12, 0, 5, tzinfo=timezone.utc),
        duration_ms=5000,
        engine="heuristic",
        total_rows=4242,
        total_tables=2,
        table_stats=[
            TableStats(table_name="users", row_count=100, rows_per_sec=2000.0),
            TableStats(table_name="orders", row_count=4142, rows_per_sec=8284.0),
        ],
    )
    snap = build_snapshot([run])
    assert snap["status"] == "complete"
    assert snap["overall_percent"] == 100
    assert snap["total_rows"] == 4242
    assert snap["total_tables"] == 2
    assert snap["completed_tables"] == 2
    names = {t["table_name"] for t in snap["tables"]}
    assert names == {"users", "orders"}


def test_build_snapshot_running_run_is_partial() -> None:
    from dbsprout.web.views.progress import build_snapshot  # noqa: PLC0415

    run = RunRecord(
        started_at=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
        completed_at=None,
        engine="heuristic",
        total_rows=0,
        total_tables=4,
        table_stats=[
            TableStats(table_name="users", row_count=100, rows_per_sec=2000.0),
            TableStats(table_name="orders", row_count=50, rows_per_sec=1000.0),
        ],
    )
    snap = build_snapshot([run])
    assert snap["status"] == "running"
    assert snap["overall_percent"] == 50
    assert snap["completed_tables"] == 2
    assert snap["current_table"] == "orders"


# ── /progress page route ──────────────────────────────────────────────


def test_progress_page_returns_200_empty_state(tmp_path: Path) -> None:
    resp = _make_client(tmp_path / "state.db").get("/progress")
    assert resp.status_code == 200
    body = resp.text
    assert 'id="nav-progress"' in body
    assert "EventSource" in body
    assert "/progress/stream" in body


def test_progress_page_returns_200_with_run(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_completed_run(state_db)
    resp = _make_client(state_db).get("/progress")
    assert resp.status_code == 200
    assert "progress" in resp.text.lower()


def test_progress_page_no_longer_placeholder(tmp_path: Path) -> None:
    """The real view replaces the S-090 'Coming soon' placeholder."""
    body = _make_client(tmp_path / "state.db").get("/progress").text
    assert "Coming soon" not in body


# ── /progress/stream SSE endpoint (BOUNDED — finite event consumption) ─


def test_stream_content_type_is_event_stream(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_completed_run(state_db)
    resp = _make_client(state_db).get("/progress/stream")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")


def test_stream_completed_run_yields_finite_events_and_sentinel(tmp_path: Path) -> None:
    """Completed run → at least one snapshot data frame + terminal sentinel.

    The generator terminates (latest run is complete), so ``.get`` buffers the
    whole finite body — no hang risk.
    """
    state_db = tmp_path / "state.db"
    _seed_completed_run(state_db)
    body = _make_client(state_db).get("/progress/stream").text

    payloads = _parse_data_frames(body)
    assert payloads, "stream must emit at least one data frame"
    assert any(p.get("status") == "complete" and p.get("overall_percent") == 100 for p in payloads)
    assert "event: complete" in body, "stream must end with a terminal sentinel"


def test_stream_empty_state_yields_idle_then_sentinel(tmp_path: Path) -> None:
    body = _make_client(tmp_path / "never.db").get("/progress/stream").text
    payloads = _parse_data_frames(body)
    assert any(p.get("status") == "idle" for p in payloads)
    assert "event: complete" in body


def test_event_generator_is_bounded(tmp_path: Path) -> None:
    """Directly drive the generator with a tiny poll cap → it MUST terminate.

    Guards against a regression to an unbounded ``while True`` stream.
    """
    from dbsprout.web.views.progress import iter_progress_events  # noqa: PLC0415

    state_db = tmp_path / "state.db"
    _seed_completed_run(state_db)
    db = StateDB(state_db)
    events = list(iter_progress_events(db.get_runs, max_polls=3, interval=0.0))
    assert len(events) <= 5, "bounded generator must emit a small, finite number of events"
    assert events[-1].startswith("event: complete"), "last frame must be the terminal sentinel"
