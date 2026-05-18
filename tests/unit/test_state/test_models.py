"""Tests for dbsprout.state.models — state-layer Pydantic models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from dbsprout.state.models import (
    LLMCall,
    QualityResult,
    RunRecord,
    TableStats,
)

_NOW = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)


class TestTableStats:
    def test_minimal(self) -> None:
        ts = TableStats(table_name="users")
        assert ts.table_name == "users"
        assert ts.id is None
        assert ts.run_id is None
        assert ts.row_count == 0
        assert ts.generation_ms == 0
        assert ts.rows_per_sec == 0.0
        assert ts.errors == 0

    def test_frozen(self) -> None:
        ts = TableStats(table_name="users")
        with pytest.raises(ValidationError):
            ts.table_name = "changed"  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden"):
            TableStats(table_name="users", bogus=1)  # type: ignore[call-arg]


class TestQualityResult:
    def test_minimal(self) -> None:
        qr = QualityResult(metric_type="integrity", metric_name="fk_valid")
        assert qr.metric_type == "integrity"
        assert qr.metric_name == "fk_valid"
        assert qr.id is None
        assert qr.run_id is None
        assert qr.score == 0.0
        assert qr.passed is False
        assert qr.details_json is None

    def test_frozen(self) -> None:
        qr = QualityResult(metric_type="fidelity", metric_name="ks")
        with pytest.raises(ValidationError):
            qr.score = 1.0  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden"):
            QualityResult(metric_type="x", metric_name="y", bogus=1)  # type: ignore[call-arg]


class TestLLMCall:
    def test_minimal(self) -> None:
        call = LLMCall(timestamp=_NOW, provider="ollama", model="qwen2.5")
        assert call.timestamp == _NOW
        assert call.provider == "ollama"
        assert call.model == "qwen2.5"
        assert call.id is None
        assert call.run_id is None
        assert call.tokens_sent == 0
        assert call.tokens_received == 0
        assert call.cost_usd == 0.0
        assert call.privacy_tier is None
        assert call.cached is False

    def test_frozen(self) -> None:
        call = LLMCall(timestamp=_NOW, provider="ollama", model="qwen2.5")
        with pytest.raises(ValidationError):
            call.model = "changed"  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden"):
            LLMCall(timestamp=_NOW, provider="x", model="y", bogus=1)  # type: ignore[call-arg]


class TestRunRecord:
    def test_minimal_defaults(self) -> None:
        run = RunRecord(started_at=_NOW, engine="heuristic")
        assert run.started_at == _NOW
        assert run.engine == "heuristic"
        assert run.id is None
        assert run.completed_at is None
        assert run.duration_ms is None
        assert run.llm_provider is None
        assert run.llm_model is None
        assert run.total_rows == 0
        assert run.total_tables == 0
        assert run.seed is None
        assert run.config_json is None
        assert run.table_stats == []
        assert run.quality_results == []
        assert run.llm_calls == []

    def test_with_children(self) -> None:
        run = RunRecord(
            started_at=_NOW,
            completed_at=_NOW,
            duration_ms=1234,
            engine="spec",
            llm_provider="ollama",
            llm_model="qwen2.5",
            total_rows=100,
            total_tables=2,
            seed=42,
            config_json='{"k": "v"}',
            table_stats=[TableStats(table_name="users", row_count=50)],
            quality_results=[QualityResult(metric_type="integrity", metric_name="fk", passed=True)],
            llm_calls=[LLMCall(timestamp=_NOW, provider="ollama", model="qwen2.5")],
        )
        assert len(run.table_stats) == 1
        assert len(run.quality_results) == 1
        assert len(run.llm_calls) == 1
        assert run.table_stats[0].table_name == "users"

    def test_frozen(self) -> None:
        run = RunRecord(started_at=_NOW, engine="heuristic")
        with pytest.raises(ValidationError):
            run.engine = "changed"  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden"):
            RunRecord(started_at=_NOW, engine="x", bogus=1)  # type: ignore[call-arg]
