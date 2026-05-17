"""Plotly-schema chart-spec + quality-table builders for the HTML report.

Pure functions: in -> :class:`RunRecord` telemetry, out -> plain dicts
that ``json.dumps`` cleanly and embed as Plotly figure specs in
``<script>`` tags. No ``plotly`` Python package is imported -- Plotly.js
is JS-only (CDN) at render time.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dbsprout.state.models import RunRecord

#: Cap charts/series to keep the report small and readable.
TOP_N = 10

#: Numeric per-table series surfaced as histograms.
_NUMERIC_SERIES: tuple[tuple[str, str], ...] = (
    ("row_count", "Row count per table"),
    ("generation_ms", "Generation time per table (ms)"),
    ("rows_per_sec", "Throughput per table (rows/sec)"),
)


def build_numeric_histograms(run: RunRecord) -> list[dict[str, Any]]:
    """Build one Plotly histogram spec per numeric per-table series."""
    stats = run.table_stats[:TOP_N]
    specs: list[dict[str, Any]] = []
    for attr, title in _NUMERIC_SERIES:
        values = [getattr(s, attr) for s in stats]
        if not values:
            continue
        specs.append(
            {
                "data": [
                    {
                        "type": "histogram",
                        "x": values,
                        "name": attr,
                        "nbinsx": min(len(values), 20),
                    }
                ],
                "layout": {
                    "title": {"text": title},
                    "bargap": 0.05,
                    "margin": {"t": 40, "r": 16, "b": 40, "l": 48},
                },
                "config": {"displaylogo": False, "responsive": True},
            }
        )
    return specs


def build_categorical_bars(run: RunRecord) -> list[dict[str, Any]]:
    """Build Plotly bar specs from any quality result whose
    ``details_json`` carries a ``{"value_counts": {label: count}}`` map.
    """
    specs: list[dict[str, Any]] = []
    for q in run.quality_results:
        if not q.details_json:
            continue
        try:
            payload = json.loads(q.details_json)
        except (ValueError, TypeError):
            continue
        counts = payload.get("value_counts") if isinstance(payload, dict) else None
        if not isinstance(counts, dict) or not counts:
            continue
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:TOP_N]
        specs.append(
            {
                "data": [
                    {
                        "type": "bar",
                        "x": [str(k) for k, _ in items],
                        "y": [v for _, v in items],
                        "name": q.metric_name,
                    }
                ],
                "layout": {
                    "title": {"text": f"Value frequency — {q.metric_name}"},
                    "margin": {"t": 40, "r": 16, "b": 60, "l": 48},
                },
                "config": {"displaylogo": False, "responsive": True},
            }
        )
        if len(specs) >= TOP_N:
            break
    return specs
