"""Generation-run telemetry writer (S-080).

Maps an in-memory generation run (:class:`~dbsprout.generate.orchestrator.
GenerateResult` + :class:`~dbsprout.quality.integrity.IntegrityReport`) to
the state-layer :class:`~dbsprout.state.models.RunRecord` and persists it
via :class:`~dbsprout.state.db.StateDB`.

The state DB is **optional telemetry**: every public failure mode is
swallowed with a warning so a state-write problem can never break or slow
``dbsprout generate`` (S-080 acceptance criteria). The write runs *after*
generation completes (outside the timed loop), so it adds no generation
overhead.

LLM-call capture is intentionally conservative: token/cost accounting is
not exposed by the embedded llama-cpp provider, so we record only the
fields that are actually knowable (provider/model/cached/privacy tier) and
leave token/cost at their model defaults rather than fabricate values.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from dbsprout.state.db import StateDB
from dbsprout.state.models import LLMCall, QualityResult, RunRecord, TableStats

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.generate.orchestrator import GenerateResult
    from dbsprout.quality.integrity import IntegrityReport

logger = logging.getLogger(__name__)

#: Default state-DB location, matching :class:`StateDB`'s own default.
_DEFAULT_DB_PATH = ".dbsprout/state.db"


def _rows_per_sec(row_count: int, generation_ms: int) -> float:
    """Rows/sec from a row count and a millisecond duration.

    Returns ``0.0`` when ``generation_ms`` is zero (sub-millisecond
    generation) to avoid a division-by-zero.
    """
    if generation_ms <= 0:
        return 0.0
    return round(row_count / (generation_ms / 1000.0), 3)


def _table_stats(result: GenerateResult) -> list[TableStats]:
    """Map per-table timings to :class:`TableStats` records."""
    return [
        TableStats(
            table_name=name,
            row_count=row_count,
            generation_ms=generation_ms,
            rows_per_sec=_rows_per_sec(row_count, generation_ms),
        )
        for name, row_count, generation_ms in result.table_timings
    ]


def _quality_results(report: IntegrityReport) -> list[QualityResult]:
    """Map integrity checks to :class:`QualityResult` records."""
    return [
        QualityResult(
            metric_type="integrity",
            metric_name=check.check,
            score=0.0,
            passed=check.passed,
            details_json=check.details or None,
        )
        for check in report.checks
    ]


def llm_call_for(
    *,
    engine: str,
    lora_path: Path | None,
    cached: bool,
) -> LLMCall | None:
    """Build an :class:`LLMCall` for the real-LLM spec path, else ``None``.

    A real LLM invocation only happens when ``engine == "spec"`` *and* a
    LoRA adapter path is supplied (the embedded provider path in
    :func:`dbsprout.generate.orchestrator._select_engines`). Without a
    LoRA the spec engine uses the deterministic heuristic fallback — no
    LLM call to record.

    Token/cost are left at their :class:`LLMCall` defaults (``0``): the
    embedded llama-cpp provider does not surface token accounting, and
    fabricating counts would be dishonest telemetry.
    """
    if engine != "spec" or lora_path is None:
        return None
    return LLMCall(
        timestamp=datetime.now(tz=timezone.utc),
        provider="embedded",
        model=lora_path.name,
        privacy_tier="local",
        cached=cached,
    )


def build_run_record(  # noqa: PLR0913
    result: GenerateResult,
    report: IntegrityReport,
    *,
    engine: str,
    seed: int,
    started_at: datetime,
    completed_at: datetime | None = None,
    config_json: str | None = None,
    llm_call: LLMCall | None = None,
) -> RunRecord:
    """Assemble a :class:`RunRecord` from an in-memory generation run."""
    duration_ms: int | None = None
    if completed_at is not None:
        duration_ms = int((completed_at - started_at).total_seconds() * 1000)

    llm_calls = [llm_call] if llm_call is not None else []
    return RunRecord(
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
        engine=engine,
        llm_provider=llm_call.provider if llm_call is not None else None,
        llm_model=llm_call.model if llm_call is not None else None,
        total_rows=result.total_rows,
        total_tables=result.total_tables,
        seed=seed,
        config_json=config_json,
        table_stats=_table_stats(result),
        quality_results=_quality_results(report),
        llm_calls=llm_calls,
    )


def record_generation_run(  # noqa: PLR0913
    result: GenerateResult,
    report: IntegrityReport,
    *,
    engine: str,
    seed: int,
    started_at: datetime,
    completed_at: datetime | None = None,
    config_json: str | None = None,
    llm_call: LLMCall | None = None,
    db_path: Path | str = _DEFAULT_DB_PATH,
) -> int | None:
    """Persist a generation run to the state DB; never raise.

    Returns the new ``runs.id`` on success, or ``None`` if anything went
    wrong (the failure is logged as a warning). Generation correctness
    and performance never depend on this call succeeding.
    """
    try:
        run = build_run_record(
            result,
            report,
            engine=engine,
            seed=seed,
            started_at=started_at,
            completed_at=completed_at,
            config_json=config_json,
            llm_call=llm_call,
        )
        return StateDB(db_path).record_run(run)
    except Exception as exc:
        logger.warning(
            "Could not record generation run to state DB (%s); "
            "continuing — state telemetry is optional.",
            exc,
        )
        return None
