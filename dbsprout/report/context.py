"""Build the view-model passed to the HTML report template.

The report module *reads* state-layer telemetry (S-079/S-080) and shapes it
into a plain ``dict`` the Jinja2 template consumes. Keeping the view-model
separate from both the state models and the template keeps the template
dumb and the state layer decoupled.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from dbsprout.report.erd import build_erd_mermaid

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.state.models import RunRecord


def _format_duration(duration_ms: int | None) -> str:
    """Render a millisecond duration as a short human string."""
    if duration_ms is None:
        return "—"
    seconds = duration_ms / 1000.0
    if seconds < 1:
        return f"{duration_ms} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s"


def _summary(run: RunRecord) -> dict[str, Any]:
    return {
        "engine": run.engine,
        "total_rows": run.total_rows,
        "total_tables": run.total_tables,
        "seed": run.seed,
        "llm_provider": run.llm_provider,
        "llm_model": run.llm_model,
        "started_at": run.started_at.isoformat(),
        "completed_at": (run.completed_at.isoformat() if run.completed_at else None),
        "duration_ms": run.duration_ms,
        "duration_human": _format_duration(run.duration_ms),
    }


def _table_stats(run: RunRecord) -> list[dict[str, Any]]:
    return [
        {
            "table_name": t.table_name,
            "row_count": t.row_count,
            "generation_ms": t.generation_ms,
            "rows_per_sec": round(t.rows_per_sec, 1),
            "errors": t.errors,
        }
        for t in run.table_stats
    ]


def _quality_results(run: RunRecord) -> list[dict[str, Any]]:
    return [
        {
            "metric_type": q.metric_type,
            "metric_name": q.metric_name,
            "score": round(q.score, 4),
            "passed": q.passed,
            "details_json": q.details_json,
        }
        for q in run.quality_results
    ]


def build_report_context(
    run: RunRecord | None,
    schema: DatabaseSchema | None = None,
) -> dict[str, Any]:
    """Shape a :class:`RunRecord` (or ``None``) into the template context.

    ``run`` is ``None`` when the state DB has no recorded runs; the template
    renders a graceful empty state in that case. ``schema`` is optional; when
    provided, a Mermaid ``erDiagram`` source string is added under
    ``erd_mermaid`` (S-082) for the ERD section to embed.
    """
    generated_at = datetime.now(timezone.utc).isoformat()
    # ─── S-082 ERD (parallel-wave region; parent reconciles) ───────────
    erd_mermaid: str | None = None
    if schema is not None:
        erd_mermaid = build_erd_mermaid(schema)
    # ─── end S-082 region ──────────────────────────────────────────────
    if run is None:
        return {
            "summary": None,
            "table_stats": [],
            "quality_results": [],
            "generated_at": generated_at,
            "erd_mermaid": erd_mermaid,
        }
    return {
        "summary": _summary(run),
        "table_stats": _table_stats(run),
        "quality_results": _quality_results(run),
        "generated_at": generated_at,
        "erd_mermaid": erd_mermaid,
    }
