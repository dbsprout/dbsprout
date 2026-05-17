"""Tests for dbsprout.state.writer — generation-run telemetry writer (S-080)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from dbsprout.generate.orchestrator import GenerateResult
from dbsprout.quality.integrity import CheckResult, IntegrityReport
from dbsprout.state import writer
from dbsprout.state.db import StateDB
from dbsprout.state.models import LLMCall
from dbsprout.state.writer import (
    build_run_record,
    llm_call_for,
    record_generation_run,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_T0 = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2026, 5, 16, 12, 0, 1, tzinfo=timezone.utc)


def _result() -> GenerateResult:
    return GenerateResult(
        tables_data={"users": [{"id": 1}], "orders": [{"id": 1}, {"id": 2}]},
        insertion_order=["users", "orders"],
        total_rows=3,
        total_tables=2,
        duration_seconds=0.5,
        table_timings=(("users", 1, 100), ("orders", 2, 0)),
    )


def _report(passed: bool = True) -> IntegrityReport:
    checks = [
        CheckResult(
            check="pk_uniqueness",
            table="users",
            column="id",
            passed=True,
            details="ok",
        ),
        CheckResult(
            check="fk_satisfaction",
            table="orders",
            column="user_id",
            passed=passed,
            details="0 violations" if passed else "1 violation",
        ),
    ]
    return IntegrityReport(checks=checks, passed=passed)


class TestBuildRunRecord:
    def test_maps_run_metadata(self) -> None:
        run = build_run_record(
            _result(),
            _report(),
            engine="heuristic",
            seed=42,
            started_at=_T0,
            completed_at=_T1,
        )

        assert run.engine == "heuristic"
        assert run.seed == 42
        assert run.started_at == _T0
        assert run.completed_at == _T1
        assert run.total_rows == 3
        assert run.total_tables == 2
        assert run.duration_ms == 1000  # _T1 - _T0 == 1s

    def test_maps_per_table_stats(self) -> None:
        run = build_run_record(_result(), _report(), engine="heuristic", seed=1, started_at=_T0)

        stats = {s.table_name: s for s in run.table_stats}
        assert set(stats) == {"users", "orders"}
        assert stats["users"].row_count == 1
        assert stats["users"].generation_ms == 100
        assert stats["users"].rows_per_sec == 10.0  # 1 row / 0.1s
        # generation_ms == 0 must not divide-by-zero
        assert stats["orders"].generation_ms == 0
        assert stats["orders"].rows_per_sec == 0.0

    def test_maps_quality_results(self) -> None:
        run = build_run_record(
            _result(), _report(passed=False), engine="heuristic", seed=1, started_at=_T0
        )

        assert len(run.quality_results) == 2
        qr = {r.metric_name: r for r in run.quality_results}
        assert qr["pk_uniqueness"].metric_type == "integrity"
        assert qr["pk_uniqueness"].passed is True
        assert qr["fk_satisfaction"].passed is False
        assert qr["fk_satisfaction"].score == 0.0


class TestLLMCallCapture:
    def test_no_llm_call_for_heuristic(self) -> None:
        assert llm_call_for(engine="heuristic", lora_path=None, cached=False) is None

    def test_no_llm_call_for_spec_without_lora(self) -> None:
        assert llm_call_for(engine="spec", lora_path=None, cached=False) is None

    def test_llm_call_for_spec_with_lora(self, tmp_path: Path) -> None:
        adapter = tmp_path / "a.gguf"
        adapter.write_bytes(b"x")
        call = llm_call_for(engine="spec", lora_path=adapter, cached=True)

        assert call is not None
        assert call.provider == "embedded"
        assert call.model == "a.gguf"
        assert call.cached is True
        assert call.privacy_tier == "local"

    def test_build_run_record_includes_llm_call(self) -> None:
        call = LLMCall(
            timestamp=_T0,
            provider="embedded",
            model="m.gguf",
            cached=False,
        )
        run = build_run_record(
            _result(),
            _report(),
            engine="spec",
            seed=1,
            started_at=_T0,
            llm_call=call,
        )
        assert len(run.llm_calls) == 1
        assert run.llm_calls[0].model == "m.gguf"
        assert run.llm_provider == "embedded"
        assert run.llm_model == "m.gguf"


class TestRecordGenerationRun:
    def test_persists_run_and_returns_id(self, tmp_path: Path) -> None:
        db_path = tmp_path / "state.db"
        run_id = record_generation_run(
            _result(),
            _report(),
            engine="heuristic",
            seed=42,
            started_at=_T0,
            completed_at=_T1,
            db_path=db_path,
        )

        assert run_id is not None
        assert run_id > 0
        runs = StateDB(db_path).get_runs()
        assert len(runs) == 1
        assert runs[0].total_rows == 3
        assert len(runs[0].table_stats) == 2

    def test_state_failure_never_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("disk full")

        monkeypatch.setattr(writer, "StateDB", _boom)

        with caplog.at_level(logging.WARNING):
            result = writer.record_generation_run(
                _result(),
                _report(),
                engine="heuristic",
                seed=1,
                started_at=_T0,
                db_path=tmp_path / "state.db",
            )

        assert result is None
        assert any("state" in r.message.lower() for r in caplog.records)
