"""Shared fixtures for report-generator tests."""

from __future__ import annotations

from datetime import datetime, timedelta

from dbsprout.state.models import LLMCall, QualityResult, RunRecord, TableStats

_T0 = datetime(2026, 5, 17, 9, 0, 0)


def make_run(offset_minutes: int = 0, engine: str = "heuristic") -> RunRecord:
    """Build a realistic :class:`RunRecord` for report tests."""
    start = _T0 + timedelta(minutes=offset_minutes)
    return RunRecord(
        started_at=start,
        completed_at=start + timedelta(seconds=5),
        duration_ms=5000,
        engine=engine,
        llm_provider="ollama",
        llm_model="qwen2.5",
        total_rows=150,
        total_tables=2,
        seed=42,
        config_json=f'{{"engine": "{engine}"}}',
        table_stats=[
            TableStats(
                table_name="users",
                row_count=100,
                generation_ms=10,
                rows_per_sec=10000.0,
            ),
            TableStats(
                table_name="orders",
                row_count=50,
                generation_ms=20,
                rows_per_sec=2500.0,
                errors=1,
            ),
        ],
        quality_results=[
            QualityResult(
                metric_type="integrity",
                metric_name="fk_valid",
                score=1.0,
                passed=True,
                details_json='{"violations": 0}',
            ),
            QualityResult(
                metric_type="fidelity",
                metric_name="distribution_match",
                score=0.92,
                passed=True,
                details_json=None,
            ),
        ],
        llm_calls=[
            LLMCall(
                timestamp=start,
                provider="ollama",
                model="qwen2.5",
                tokens_sent=200,
                tokens_received=80,
                cost_usd=0.0,
                privacy_tier="local",
                cached=True,
            )
        ],
    )
