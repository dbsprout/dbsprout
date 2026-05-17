"""Unit tests for the report view-model builder (Task 2)."""

from __future__ import annotations

from dbsprout.report.context import build_report_context

from ._fixtures import make_run


class TestBuildReportContext:
    def test_run_summary_fields(self) -> None:
        ctx = build_report_context(make_run())
        summary = ctx["summary"]
        assert summary["engine"] == "heuristic"
        assert summary["total_rows"] == 150
        assert summary["total_tables"] == 2
        assert summary["seed"] == 42
        assert summary["llm_provider"] == "ollama"
        assert summary["duration_human"]  # non-empty formatted duration
        assert summary["started_at"]  # ISO-ish timestamp string

    def test_table_stats_rows(self) -> None:
        ctx = build_report_context(make_run())
        tables = ctx["table_stats"]
        assert len(tables) == 2
        names = {t["table_name"] for t in tables}
        assert names == {"users", "orders"}
        orders = next(t for t in tables if t["table_name"] == "orders")
        assert orders["row_count"] == 50
        assert orders["errors"] == 1

    def test_quality_metrics_rows(self) -> None:
        ctx = build_report_context(make_run())
        quality = ctx["quality_results"]
        assert len(quality) == 2
        assert any(q["metric_name"] == "fk_valid" and q["passed"] for q in quality)

    def test_generated_at_present(self) -> None:
        ctx = build_report_context(make_run())
        assert ctx["generated_at"]

    def test_handles_none_run_empty_state(self) -> None:
        ctx = build_report_context(None)
        assert ctx["summary"] is None
        assert ctx["table_stats"] == []
        assert ctx["quality_results"] == []
        assert ctx["generated_at"]

    def test_handles_run_with_no_children(self) -> None:
        run = make_run().model_copy(
            update={"table_stats": [], "quality_results": [], "llm_calls": []}
        )
        ctx = build_report_context(run)
        assert ctx["summary"]["engine"] == "heuristic"
        assert ctx["table_stats"] == []
        assert ctx["quality_results"] == []
