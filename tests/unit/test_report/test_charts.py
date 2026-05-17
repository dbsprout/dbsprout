"""Unit tests for report chart-spec + quality-table builders (S-083)."""

from __future__ import annotations

import json

from dbsprout.report.charts import (
    build_categorical_bars,
    build_correlation_heatmap,
    build_numeric_histograms,
)
from dbsprout.state.models import QualityResult

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


class TestCategoricalBars:
    def test_value_frequency_from_details_json(self) -> None:
        run = make_run().model_copy(
            update={
                "quality_results": [
                    QualityResult(
                        metric_type="distribution",
                        metric_name="users.status",
                        score=1.0,
                        passed=True,
                        details_json='{"value_counts": {"active": 80, "inactive": 20}}',
                    )
                ]
            }
        )
        specs = build_categorical_bars(run)
        assert specs
        bar = specs[0]["data"][0]
        assert bar["type"] == "bar"
        assert set(bar["x"]) == {"active", "inactive"}
        assert dict(zip(bar["x"], bar["y"], strict=True))["active"] == 80

    def test_ignores_results_without_value_counts(self) -> None:
        assert build_categorical_bars(make_run()) == []


class TestCorrelationHeatmap:
    def test_heatmap_from_correlation_payload(self) -> None:
        run = make_run().model_copy(
            update={
                "quality_results": [
                    QualityResult(
                        metric_type="fidelity",
                        metric_name="correlation",
                        score=0.9,
                        passed=True,
                        details_json=(
                            '{"correlation": {"labels": ["a", "b"], '
                            '"matrix": [[1.0, 0.5], [0.5, 1.0]]}}'
                        ),
                    )
                ]
            }
        )
        spec = build_correlation_heatmap(run)
        assert spec is not None
        assert spec["data"][0]["type"] == "heatmap"
        assert spec["data"][0]["z"] == [[1.0, 0.5], [0.5, 1.0]]

    def test_none_when_no_fidelity_correlation(self) -> None:
        assert build_correlation_heatmap(make_run()) is None
