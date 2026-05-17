"""Plotly-schema chart-spec + quality-table builders for the HTML report.

Pure functions: in -> :class:`RunRecord` telemetry, out -> plain dicts
that ``json.dumps`` cleanly and embed as Plotly figure specs in
``<script>`` tags. No ``plotly`` Python package is imported -- Plotly.js
is JS-only (CDN) at render time.
"""

from __future__ import annotations

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
