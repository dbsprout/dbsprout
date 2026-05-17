"""Unit tests for report chart-spec + quality-table builders (S-083)."""

from __future__ import annotations

import json

from dbsprout.report.charts import build_numeric_histograms

from ._fixtures import make_run


class TestNumericHistograms:
    def test_one_trace_per_numeric_series(self) -> None:
        specs = build_numeric_histograms(make_run())
        assert specs, "expected at least one numeric histogram"
        spec = specs[0]
        assert spec["data"][0]["type"] == "histogram"
        # Spec is JSON-serialisable (embedded in <script>).
        json.dumps(spec)

    def test_titles_reference_known_metrics(self) -> None:
        titles = {s["layout"]["title"]["text"] for s in build_numeric_histograms(make_run())}
        assert any("row" in t.lower() for t in titles)

    def test_top_ten_cap(self) -> None:
        run = make_run()
        stats = run.table_stats[0]
        many = [stats.model_copy(update={"table_name": f"t{i}"}) for i in range(25)]
        run = run.model_copy(update={"table_stats": many})
        specs = build_numeric_histograms(run)
        for spec in specs:
            assert len(spec["data"][0]["x"]) <= 10
