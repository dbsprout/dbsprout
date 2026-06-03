"""JSON insights API tests (P1c-4): /api/runs, /api/quality, /api/costs.

These three read-only JSON endpoints wrap the existing pure builders
(``paginate_runs``, ``build_quality_table``, ``build_cost_summary``) over the
SQLite state layer (``.dbsprout/state.db``). They feed the React Workbench's
"Runs & Quality" panels (the HTML ``views/insights.py`` stays until P1c-5).

The web stack lives in the optional ``[web]`` extra, so every test guards with
``pytest.importorskip("fastapi")`` *before* importing FastAPI symbols (mirrors
``tests/test_web/test_insights_views.py``). Each endpoint is exercised with a
``TestClient`` over a temporary state DB, plus an empty-state case that must
return 200 (never 500) when the CLI has never run — honest "no data".
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.state.db import StateDB
from dbsprout.state.models import LLMCall, QualityResult, RunRecord, TableStats

if TYPE_CHECKING:
    from pathlib import Path


def _make_client(state_db: Path) -> TestClient:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return TestClient(create_app(state_db_path=state_db))


def _seed_run(
    state_db: Path,
    *,
    started_minute: int = 0,
    engine: str = "spec",
    with_llm: bool = True,
) -> int:
    """Insert a populated run (table_stats + quality + optional llm_calls)."""
    db = StateDB(state_db)
    llm_calls = (
        [
            LLMCall(
                timestamp=datetime(2026, 5, 20, 12, started_minute, tzinfo=timezone.utc),
                provider="openai",
                model="gpt-4o-mini",
                tokens_sent=1200,
                tokens_received=800,
                cost_usd=0.05,
            ),
            LLMCall(
                timestamp=datetime(2026, 5, 20, 12, started_minute, 30, tzinfo=timezone.utc),
                provider="anthropic",
                model="claude-3-haiku",
                tokens_sent=600,
                tokens_received=400,
                cost_usd=0.03,
            ),
        ]
        if with_llm
        else []
    )
    return db.record_run(
        RunRecord(
            started_at=datetime(2026, 5, 20, 12, started_minute, tzinfo=timezone.utc),
            completed_at=datetime(2026, 5, 20, 12, started_minute, 5, tzinfo=timezone.utc),
            duration_ms=5000,
            engine=engine,
            llm_provider="openai" if with_llm else None,
            llm_model="gpt-4o-mini" if with_llm else None,
            total_rows=4242,
            total_tables=2,
            seed=42,
            config_json='{"secret":"should-not-leak"}',
            table_stats=[
                TableStats(
                    table_name="users", row_count=100, generation_ms=12, rows_per_sec=8333.0
                ),
                TableStats(
                    table_name="orders", row_count=4142, generation_ms=40, rows_per_sec=10355.0
                ),
            ],
            quality_results=[
                QualityResult(
                    metric_type="integrity", metric_name="fk_valid", score=1.0, passed=True
                ),
                QualityResult(
                    metric_type="fidelity", metric_name="distribution", score=0.62, passed=True
                ),
                QualityResult(
                    metric_type="detection", metric_name="classifier", score=0.4, passed=False
                ),
            ],
            llm_calls=llm_calls,
        )
    )


# ── /api/runs ──────────────────────────────────────────────────────────────


def test_runs_returns_history_rows(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_run(state_db)
    resp = _make_client(state_db).get("/api/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_runs"] == 1
    assert body["page"] == 1
    assert len(body["rows"]) == 1
    row = body["rows"][0]
    assert row["engine"] == "spec"
    assert row["provider"] == "openai"
    assert row["total_rows"] == 4242
    assert row["total_tables"] == 2
    assert row["cost"] == pytest.approx(0.08)
    assert "started_at" in row


def test_runs_paginates(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    for minute in range(15):
        _seed_run(state_db, started_minute=minute, with_llm=False)
    client = _make_client(state_db)
    page1 = client.get("/api/runs").json()
    assert page1["total_runs"] == 15
    assert page1["total_pages"] == 2
    assert len(page1["rows"]) == 10
    assert page1["has_prev"] is False
    assert page1["has_next"] is True
    page2 = client.get("/api/runs", params={"page": 2}).json()
    assert page2["page"] == 2
    assert len(page2["rows"]) == 5
    assert page2["has_prev"] is True
    assert page2["has_next"] is False


def test_runs_clamps_out_of_range_page(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_run(state_db, with_llm=False)
    resp = _make_client(state_db).get("/api/runs", params={"page": 999})
    assert resp.status_code == 200
    assert resp.json()["page"] == 1


def test_runs_empty_state(tmp_path: Path) -> None:
    resp = _make_client(tmp_path / "empty.db").get("/api/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_runs"] == 0
    assert body["rows"] == []
    assert body["total_pages"] == 1
    assert body["has_prev"] is False
    assert body["has_next"] is False


def test_runs_does_not_leak_config_json(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_run(state_db)
    body = _make_client(state_db).get("/api/runs").text
    assert "should-not-leak" not in body
    assert "config_json" not in body


# ── /api/quality ─────────────────────────────────────────────────────────────


def test_quality_latest_run_statuses(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    run_id = _seed_run(state_db)
    resp = _make_client(state_db).get("/api/quality")
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["run_id"] == run_id
    statuses = {r["metric_name"]: r["status"] for r in body["rows"]}
    assert statuses["fk_valid"] == "pass"
    assert statuses["distribution"] == "warn"  # fidelity 0.62 < 0.8
    assert statuses["classifier"] == "fail"  # not passed


def test_quality_by_run_id(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    first = _seed_run(state_db, started_minute=0, engine="heuristic", with_llm=False)
    _seed_run(state_db, started_minute=5, engine="spec")
    resp = _make_client(state_db).get("/api/quality", params={"run_id": first})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["run_id"] == first


def test_quality_unknown_run_id_empty(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_run(state_db)
    resp = _make_client(state_db).get("/api/quality", params={"run_id": 999999})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert body["rows"] == []


def test_quality_empty_state(tmp_path: Path) -> None:
    resp = _make_client(tmp_path / "empty.db").get("/api/quality")
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert body["rows"] == []
    assert body["run_id"] is None


# ── /api/costs ───────────────────────────────────────────────────────────────


def test_costs_aggregates_totals_and_providers(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_run(state_db)
    resp = _make_client(state_db).get("/api/costs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_cost"] == pytest.approx(0.08)
    assert body["total_tokens"] == 3000  # 1200+800+600+400
    assert body["total_calls"] == 2
    assert body["avg_cost_per_run"] == pytest.approx(0.08)
    providers = {p["provider"]: p for p in body["per_provider"]}
    assert providers["openai"]["cost"] == pytest.approx(0.05)
    assert providers["anthropic"]["cost"] == pytest.approx(0.03)
    # per-provider sorted by descending cost
    assert body["per_provider"][0]["provider"] == "openai"


def test_costs_no_llm_calls(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_run(state_db, with_llm=False)
    resp = _make_client(state_db).get("/api/costs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_cost"] == 0.0
    assert body["total_calls"] == 0
    assert body["per_provider"] == []


def test_costs_empty_state(tmp_path: Path) -> None:
    resp = _make_client(tmp_path / "empty.db").get("/api/costs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_cost"] == 0.0
    assert body["per_provider"] == []


def test_costs_does_not_expose_chart(tmp_path: Path) -> None:
    """The SPA renders its own chart; the JSON endpoint omits the Plotly spec."""
    state_db = tmp_path / "state.db"
    _seed_run(state_db)
    body = _make_client(state_db).get("/api/costs").json()
    assert "chart" not in body


# ── pure row builder ────────────────────────────────────────────────────────


def test_run_row_sums_cost_from_llm_calls() -> None:
    from dbsprout.web.routers.insights_api import build_run_row  # noqa: PLC0415

    run = RunRecord(
        id=7,
        started_at=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
        engine="spec",
        llm_provider="openai",
        total_rows=10,
        total_tables=1,
        duration_ms=1234,
        llm_calls=[
            LLMCall(
                timestamp=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
                provider="openai",
                model="gpt-4o",
                cost_usd=0.10,
            ),
            LLMCall(
                timestamp=datetime(2026, 5, 20, 12, 1, tzinfo=timezone.utc),
                provider="openai",
                model="gpt-4o",
                cost_usd=0.20,
            ),
        ],
    )
    row = build_run_row(run)
    assert row.id == 7
    assert row.engine == "spec"
    assert row.provider == "openai"
    assert row.total_rows == 10
    assert row.duration_ms == 1234
    assert row.cost == pytest.approx(0.30)


def test_run_row_provider_dash_when_none() -> None:
    from dbsprout.web.routers.insights_api import build_run_row  # noqa: PLC0415

    run = RunRecord(
        started_at=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
        engine="heuristic",
    )
    row = build_run_row(run)
    assert row.provider is None
    assert row.cost == 0.0
