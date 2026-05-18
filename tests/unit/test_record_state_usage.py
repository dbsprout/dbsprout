"""S-080a: _record_state threads GenerateResult.spec_usage into the LLMCall."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from dbsprout.cli.commands.generate import _record_state
from dbsprout.generate.orchestrator import GenerateResult
from dbsprout.quality.integrity import IntegrityReport
from dbsprout.spec.providers.base import SpecUsage


def _report() -> IntegrityReport:
    return IntegrityReport(checks=[])


class TestRecordStateUsageThreading:
    def test_spec_usage_threaded_into_llm_call(self) -> None:
        """The real usage on GenerateResult reaches llm_call_for(usage=...)."""
        usage = SpecUsage(tokens_sent=321, tokens_received=654, cost_usd=0.0012)
        result = GenerateResult(
            total_rows=3, total_tables=1, duration_seconds=0.01, spec_usage=usage
        )
        lora = Path("/tmp/a.gguf")  # noqa: S108 — mock boundary, never read

        with (
            patch("dbsprout.state.writer.record_generation_run") as m_record,
            patch("dbsprout.state.writer.llm_call_for") as m_call_for,
        ):
            _record_state(result, _report(), engine="spec", seed=1, lora_path=lora)

        m_call_for.assert_called_once()
        _, kwargs = m_call_for.call_args
        assert kwargs["usage"] == usage
        assert kwargs["engine"] == "spec"
        assert kwargs["lora_path"] == lora
        m_record.assert_called_once()

    def test_no_usage_passes_none(self) -> None:
        """Heuristic path (spec_usage None) → usage=None, honest defaults."""
        result = GenerateResult(total_rows=1, total_tables=1, duration_seconds=0.0)

        with (
            patch("dbsprout.state.writer.record_generation_run"),
            patch("dbsprout.state.writer.llm_call_for") as m_call_for,
        ):
            _record_state(result, _report(), engine="heuristic", seed=1, lora_path=None)

        _, kwargs = m_call_for.call_args
        assert kwargs["usage"] is None
