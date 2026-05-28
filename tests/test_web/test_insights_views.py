"""Web insights views tests (S-093): quality, preview, costs, history.

These four data-backed views read exclusively from the SQLite state layer
(``.dbsprout/state.db``, S-079/S-080) — the web dashboard never imports
generation code. The web stack lives in the optional ``[web]`` extra, so every
test guards with ``pytest.importorskip("fastapi")`` *before* importing FastAPI
symbols (mirrors ``tests/test_web/test_app.py``). Each route is exercised with a
``TestClient`` over a temporary state DB, plus an empty-state case that must
return 200 (never 500) when the CLI has never run.
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


# ── pure builders ──────────────────────────────────────────────────────


def test_build_cost_summary_aggregates_across_runs() -> None:
    from dbsprout.web.views.insights import build_cost_summary  # noqa: PLC0415

    state_db_runs = [
        RunRecord(
            started_at=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
            engine="spec",
            llm_calls=[
                LLMCall(
                    timestamp=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
                    provider="openai",
                    model="gpt-4o",
                    tokens_sent=100,
                    tokens_received=50,
                    cost_usd=0.10,
                ),
                LLMCall(
                    timestamp=datetime(2026, 5, 20, 12, 1, tzinfo=timezone.utc),
                    provider="openai",
                    model="gpt-4o",
                    tokens_sent=100,
                    tokens_received=50,
                    cost_usd=0.20,
                ),
            ],
        ),
        RunRecord(
            started_at=datetime(2026, 5, 20, 13, 0, tzinfo=timezone.utc),
            engine="spec",
            llm_calls=[
                LLMCall(
                    timestamp=datetime(2026, 5, 20, 13, 0, tzinfo=timezone.utc),
                    provider="anthropic",
                    model="claude",
                    tokens_sent=200,
                    tokens_received=100,
                    cost_usd=0.30,
                ),
            ],
        ),
    ]
    summary = build_cost_summary(state_db_runs)
    assert summary["total_cost"] == pytest.approx(0.60)
    assert summary["total_tokens"] == 600
    assert summary["total_calls"] == 3
    assert summary["avg_cost_per_run"] == pytest.approx(0.30)
    providers = {p["provider"]: p for p in summary["per_provider"]}
    assert providers["openai"]["cost"] == pytest.approx(0.30)
    assert providers["anthropic"]["cost"] == pytest.approx(0.30)
    chart = summary["chart"]
    assert chart["data"][0]["type"] in {"scatter", "bar"}
    assert len(chart["data"][0]["x"]) == len(chart["data"][0]["y"]) == 2


def test_build_cost_summary_empty_runs() -> None:
    from dbsprout.web.views.insights import build_cost_summary  # noqa: PLC0415

    summary = build_cost_summary([])
    assert summary["total_cost"] == 0.0
    assert summary["total_tokens"] == 0
    assert summary["avg_cost_per_run"] == 0.0
    assert summary["per_provider"] == []
    assert summary["chart"] is None


def test_paginate_runs_slices_and_metadata() -> None:
    from dbsprout.web.views.insights import paginate_runs  # noqa: PLC0415

    runs = [
        RunRecord(started_at=datetime(2026, 5, 20, 12, i, tzinfo=timezone.utc), engine="spec")
        for i in range(25)
    ]
    page = paginate_runs(runs, page=2, per_page=10)
    assert len(page["runs"]) == 10
    assert page["page"] == 2
    assert page["total_pages"] == 3
    assert page["has_prev"] is True
    assert page["has_next"] is True


def test_paginate_runs_clamps_out_of_range() -> None:
    from dbsprout.web.views.insights import paginate_runs  # noqa: PLC0415

    runs = [
        RunRecord(started_at=datetime(2026, 5, 20, 12, i, tzinfo=timezone.utc), engine="spec")
        for i in range(5)
    ]
    assert paginate_runs(runs, page=99, per_page=10)["page"] == 1
    assert paginate_runs(runs, page=0, per_page=10)["page"] == 1
    assert paginate_runs(runs, page=-3, per_page=10)["page"] == 1


def test_paginate_runs_empty() -> None:
    from dbsprout.web.views.insights import paginate_runs  # noqa: PLC0415

    page = paginate_runs([], page=1, per_page=10)
    assert page["runs"] == []
    assert page["total_pages"] == 1
    assert page["has_prev"] is False
    assert page["has_next"] is False


# ── /quality ─────────────────────────────────────────────────────────────


def test_quality_route_renders_metrics(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_run(state_db)
    resp = _make_client(state_db).get("/quality")
    assert resp.status_code == 200
    body = resp.text
    assert "fk_valid" in body
    assert "distribution" in body
    assert "classifier" in body
    assert 'id="nav-quality"' in body


def test_quality_route_shadows_placeholder(tmp_path: Path) -> None:
    """The real /quality view replaces the S-090 'Coming soon' placeholder."""
    state_db = tmp_path / "state.db"
    _seed_run(state_db)
    body = _make_client(state_db).get("/quality").text
    assert "Coming soon" not in body


def test_quality_route_empty_state(tmp_path: Path) -> None:
    resp = _make_client(tmp_path / "empty.db").get("/quality")
    assert resp.status_code == 200
    assert "No quality" in resp.text or "No run" in resp.text


# ── /preview ─────────────────────────────────────────────────────────────


def test_preview_landing_lists_tables(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_run(state_db)
    resp = _make_client(state_db).get("/preview")
    assert resp.status_code == 200
    body = resp.text
    assert "users" in body
    assert "orders" in body


def test_preview_table_shows_stats_and_note(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_run(state_db)
    resp = _make_client(state_db).get("/preview/users")
    assert resp.status_code == 200
    body = resp.text
    assert "users" in body
    assert "100" in body  # row_count
    assert "sample rows" in body.lower()  # honest "not persisted" note


def test_preview_unknown_table_graceful(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_run(state_db)
    resp = _make_client(state_db).get("/preview/nonexistent")
    assert resp.status_code == 200
    assert "nonexistent" in resp.text


def test_preview_empty_state(tmp_path: Path) -> None:
    resp = _make_client(tmp_path / "empty.db").get("/preview")
    assert resp.status_code == 200
    assert "No run" in resp.text or "No table" in resp.text


# ── /costs ───────────────────────────────────────────────────────────────


def test_costs_route_renders_summary_and_chart(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_run(state_db)
    resp = _make_client(state_db).get("/costs")
    assert resp.status_code == 200
    body = resp.text
    assert "0.08" in body  # total cost 0.05 + 0.03
    assert "openai" in body
    assert "anthropic" in body
    assert "plotly" in body.lower()  # cost-over-time chart via CDN


def test_costs_route_no_llm_calls(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_run(state_db, with_llm=False)
    resp = _make_client(state_db).get("/costs")
    assert resp.status_code == 200
    body = resp.text
    assert "0.00" in body
    assert "No LLM" in body or "no LLM" in body


def test_costs_route_empty_state(tmp_path: Path) -> None:
    resp = _make_client(tmp_path / "empty.db").get("/costs")
    assert resp.status_code == 200


# ── /history ─────────────────────────────────────────────────────────────


def test_history_route_renders_runs(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_run(state_db)
    resp = _make_client(state_db).get("/history")
    assert resp.status_code == 200
    body = resp.text
    assert "spec" in body
    assert "4,242" in body or "4242" in body
    assert "0.08" in body  # cost-per-run from llm_calls


def test_history_route_paginates(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    for minute in range(15):
        _seed_run(state_db, started_minute=minute, with_llm=False)
    client = _make_client(state_db)
    page1 = client.get("/history")
    assert page1.status_code == 200
    page2 = client.get("/history", params={"page": 2})
    assert page2.status_code == 200
    # 15 runs / 10-per-page → page 2 exists with the remaining 5
    assert "page=1" in page2.text or "Previous" in page2.text


def test_history_route_clamps_page(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_run(state_db, with_llm=False)
    resp = _make_client(state_db).get("/history", params={"page": 999})
    assert resp.status_code == 200


def test_history_route_empty_state(tmp_path: Path) -> None:
    resp = _make_client(tmp_path / "empty.db").get("/history")
    assert resp.status_code == 200
    assert "No run" in resp.text
