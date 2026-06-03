"""``POST /api/validate`` — integrity + fidelity + detection report (S-133, S-134).

After a generation run finishes (``POST /api/generate``, S-124), the rows live in
``app.state.workspace.last_result`` (S-111) and the schema lives in
``app.state.workspace.schema``. This module surfaces an integrity report for the
last run by reusing the existing :func:`dbsprout.quality.integrity.validate_integrity`
validator (FK satisfaction, PK / UNIQUE, NOT NULL); the CHECK-constraint slot is
preserved in the envelope for future quality work (currently always ``0``).

S-134 additionally surfaces two distribution-quality blocks in the same
envelope: ``fidelity`` (KS / TV / cardinality / correlation similarity, via
:func:`dbsprout.quality.fidelity.validate_fidelity`) and ``detection`` (C2ST
classifier accuracy, via :func:`dbsprout.quality.detection.validate_detection`).
Both blocks are computed only when reference rows are available on
``app.state.workspace`` (see :meth:`~dbsprout.web.workspace.Workspace.get_reference_data`);
otherwise both keys are present in the envelope but ``None`` — the contract is
"keys always present, payload null when not computable", which keeps the JSON
shape stable for clients. The optional ``[stats]`` extra (``scipy`` for
fidelity, ``scikit-learn`` for detection) is handled the same way: if the
helper raises ``ImportError``, the affected block degrades to ``None`` and the
endpoint still returns ``200``.

The endpoint returns a JSON envelope ``{summary, by_table, details, fidelity,
detection}`` (JSON-only since the P1c-5 cutover; the legacy HTMX
``_validate_panel.html`` fragment was removed with the rest of the
server-rendered UI).

When no run has happened yet the endpoint returns ``409`` with
``{"code": "NO_RUN", "message": ...}``.

``details`` is capped at :data:`_MAX_DETAIL_ROWS` (500) — the cap protects the
client from pathological multi-thousand-violation payloads while remaining
generous enough for any realistic run. The cap is documented in the AC and
echoed in the HTML fragment header so the user can tell when they're seeing a
truncated view.

This module is imported only by :mod:`dbsprout.web.app` (itself lazy-imported by
``dbsprout serve``). It must stay import-light — the integrity validator is
lazy-imported inside the handler so importing the router stays cheap. The
:class:`~dbsprout.web.workspace.Workspace`, the
:class:`~dbsprout.generate.orchestrator.GenerateResult`, and the schema are
accessed via ``app.state`` at request time, never imported at module level
(kept under :data:`typing.TYPE_CHECKING`), preserving the ``dbsprout serve``
lazy-import contract — see
``test_validate_router_no_eager_generation_import``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Request, status

if TYPE_CHECKING:
    from dbsprout.quality.integrity import IntegrityReport
    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.web.workspace import Workspace

_log = logging.getLogger(__name__)

validate_router = APIRouter()

#: Max rows in the JSON / HTML ``details`` list. Documented in the AC.
_MAX_DETAIL_ROWS: int = 500

#: HTTP body for the NO_RUN guard. Pluggable into both the JSON envelope and
#: the HTML fragment so the message stays in one place.
_NO_RUN_DETAIL: dict[str, str] = {
    "code": "NO_RUN",
    "message": ("No generation result is available. Run POST /api/generate before validating."),
}


def _workspace(request: Request) -> Workspace:
    """Typed accessor for the shared session workspace wired in ``app.py``."""
    return cast("Workspace", request.app.state.workspace)


# ── aggregation ────────────────────────────────────────────────────────


#: Map ``IntegrityReport`` check names → JSON envelope bucket keys. ``check``
#: constraints are not currently emitted by ``validate_integrity`` but the
#: bucket exists so the envelope shape is stable across future quality work.
_CHECK_BUCKET: dict[str, str] = {
    "fk_satisfaction": "fk_violations",
    "unique": "unique_violations",
    "pk_uniqueness": "unique_violations",
    "not_null": "not_null_violations",
    "check_constraint": "check_violations",
}


def _aggregate_report(
    report: IntegrityReport,
    tables_data: dict[str, list[dict[str, Any]]],
    table_names: list[str],
) -> dict[str, Any]:
    """Shape an :class:`IntegrityReport` into the JSON envelope.

    Returns a dict with three keys: ``summary``, ``by_table``, ``details``.
    ``summary.violations`` counts *failed* checks (not bad rows — the underlying
    validator already aggregates per-table-per-constraint). ``by_table`` lists
    one row per table present in ``tables_data`` (in input order) with zeros for
    tables that passed every check; ``details`` is the (capped) list of failed
    check rows, ordered by table then check.
    """
    # Build empty per-table buckets first so tables with zero violations still
    # appear (clients render "all green" rows for them).
    buckets: dict[str, dict[str, int]] = {
        name: {
            "fk_violations": 0,
            "unique_violations": 0,
            "not_null_violations": 0,
            "check_violations": 0,
        }
        for name in table_names
    }

    details: list[dict[str, Any]] = []
    total_violations = 0
    for check in report.checks:
        if check.passed:
            continue
        bucket_key = _CHECK_BUCKET.get(check.check)
        if bucket_key is None:  # pragma: no cover — defensive: validator only emits known names
            continue
        # ``buckets`` is keyed by table_names. Tables with rows but absent from
        # the schema shouldn't happen (the validator only checks schema tables),
        # but guard defensively rather than KeyError.
        if check.table in buckets:  # pragma: no branch — validator only emits schema tables
            buckets[check.table][bucket_key] += 1
        total_violations += 1
        details.append(
            {
                "check": check.check,
                "table": check.table,
                "column": check.column,
                "passed": check.passed,
                "details": check.details,
            }
        )

    by_table = [{"table": name, **buckets[name]} for name in table_names]
    return {
        "summary": {
            "tables": len(table_names),
            "rows": sum(len(rows) for rows in tables_data.values()),
            "violations": total_violations,
        },
        "by_table": by_table,
        "details": details[:_MAX_DETAIL_ROWS],
    }


# region: fidelity+detection (S-134) ────────────────────────────────────


def _serialise_fidelity(
    synthetic: dict[str, list[dict[str, Any]]],
    reference: dict[str, list[dict[str, Any]]],
    schema: DatabaseSchema,
) -> dict[str, Any] | None:
    """Run :func:`validate_fidelity` and shape its report into a JSON-safe dict.

    Returns ``None`` when the optional ``[stats]`` extra (``scipy``) is absent —
    the route uses that to set ``body['fidelity'] = None`` without 500'ing.
    Lazy-imported so the router import stays cheap (preserves the
    ``dbsprout serve`` lazy-import contract).
    """
    try:
        from dbsprout.quality.fidelity import validate_fidelity  # noqa: PLC0415
    except ImportError:  # pragma: no cover — defensive: helper is always importable
        return None
    try:
        report = validate_fidelity(synthetic, reference, schema)
    except ImportError:
        # scipy missing → graceful-degrade to ``None``. Logged at WARNING so
        # operators notice the missing extra without an HTTP 500.
        _log.warning("fidelity skipped: scipy missing ([stats] extra)")
        return None
    return {
        "overall_score": report.overall_score,
        "passed": report.passed,
        "metrics": [
            {
                "metric": m.metric,
                "table": m.table,
                "column": m.column,
                "score": m.score,
                "details": m.details,
            }
            for m in report.metrics
        ],
    }


def _serialise_detection(
    synthetic: dict[str, list[dict[str, Any]]],
    reference: dict[str, list[dict[str, Any]]],
    schema: DatabaseSchema,
) -> dict[str, Any] | None:
    """Run :func:`validate_detection` and shape its report into a JSON-safe dict.

    Returns ``None`` when the optional ``[stats]`` extra (``scikit-learn``) is
    absent — same graceful-degrade pattern as :func:`_serialise_fidelity`.
    """
    try:
        from dbsprout.quality.detection import validate_detection  # noqa: PLC0415
    except ImportError:  # pragma: no cover — defensive: helper is always importable
        return None
    try:
        report = validate_detection(synthetic, reference, schema)
    except ImportError:
        _log.warning("detection skipped: scikit-learn missing ([stats] extra)")
        return None
    return {
        "overall_score": report.overall_score,
        "passed": report.passed,
        "metrics": [
            {
                "metric": m.metric,
                "table": m.table,
                "accuracy": m.accuracy,
                "details": m.details,
            }
            for m in report.metrics
        ],
    }


# endregion: fidelity+detection (S-134) ──────────────────────────────────


# ── handler ────────────────────────────────────────────────────────────


@validate_router.post("/api/validate", response_model=None)
async def validate_run(request: Request) -> dict[str, Any]:
    """Validate the last generation run and return an integrity report.

    Returns ``200`` with ``{summary, by_table, details, fidelity, detection}``;
    ``409`` with ``{"code": "NO_RUN", "message": ...}`` when no run is available.
    JSON-only since the P1c-5 cutover.
    """
    workspace = _workspace(request)
    last_result = workspace.get_last_result()
    schema = workspace.get_schema()

    if last_result is None or schema is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_NO_RUN_DETAIL,
        )

    # Lazy-import the validator so importing the router stays cheap (preserves
    # the ``dbsprout serve`` lazy-import contract).
    from dbsprout.quality.integrity import validate_integrity  # noqa: PLC0415

    tables_data = last_result.tables_data
    table_names = [t.name for t in schema.tables]
    report = validate_integrity(tables_data, schema)
    envelope = _aggregate_report(report, tables_data, table_names)

    # S-134: fidelity + detection — keys always present (``None`` when no
    # reference rows are seeded, ``None`` when the optional ``[stats]`` extra
    # is missing). Both helpers are lazy-imported inside ``_serialise_*``.
    reference_data = workspace.get_reference_data()
    fidelity_block: dict[str, Any] | None = None
    detection_block: dict[str, Any] | None = None
    if reference_data is not None:
        fidelity_block = _serialise_fidelity(tables_data, reference_data, schema)
        detection_block = _serialise_detection(tables_data, reference_data, schema)
    envelope["fidelity"] = fidelity_block
    envelope["detection"] = detection_block

    return envelope
