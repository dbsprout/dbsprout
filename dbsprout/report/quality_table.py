"""Quality-metrics table view-model for the HTML report (S-083).

Classifies each :class:`QualityResult` into a ``pass`` / ``fail`` /
``warn`` status so the template can render conditional formatting that
mirrors ``dbsprout validate`` terminal output (integrity pass/fail,
fidelity score thresholds).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dbsprout.state.models import RunRecord

#: Fidelity scores at or above this are healthy; below -> warn.
_FIDELITY_WARN_THRESHOLD = 0.8


def _status(metric_type: str, score: float, passed: bool) -> str:
    if not passed:
        return "fail"
    if metric_type == "fidelity" and score < _FIDELITY_WARN_THRESHOLD:
        return "warn"
    return "pass"


def build_quality_table(run: RunRecord) -> list[dict[str, Any]]:
    """Shape quality results into rows with a classified ``status``."""
    return [
        {
            "metric_type": q.metric_type,
            "metric_name": q.metric_name,
            "score": round(q.score, 4),
            "passed": q.passed,
            "status": _status(q.metric_type, q.score, q.passed),
            "details_json": q.details_json,
        }
        for q in run.quality_results
    ]
