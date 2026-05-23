"""TUI Quality screen (S-089).

Renders quality / integrity check results in the Quality tab of
:class:`~dbsprout.tui.app.DBSproutApp`:

* a summary line (run engine / total checks / pass-fail counts), and
* a :class:`~textual.widgets.DataTable` (Metric Type / Metric Name / Score /
  Status / Details) with colour-coded status cells — green ``PASS``,
  red ``FAIL``, yellow ``WARN``.

**Data source.** Read-only of the shared state DB (``.dbsprout/state.db``) via
:class:`~dbsprout.state.db.StateDB`. The CLI ``generate``/``validate`` flow
writes a :class:`~dbsprout.state.models.QualityResult` per check (today only
integrity checks; fidelity/detection rows share the same contract and render
if present). The screen shows the **newest** run's results on mount.

**Status classification** mirrors the HTML report
(:func:`dbsprout.report.quality_table.build_quality_table`) so the TUI and the
report never disagree: ``fail`` when a check did not pass, ``warn`` for a
fidelity metric whose score is below the healthy threshold, otherwise ``pass``.
Integrity rows carry ``score=0.0`` and therefore never spuriously ``warn``.

``textual`` is the optional ``[tui]`` extra. This module is imported lazily
(only by the running app), so the top-level textual import never touches the
hot CLI-startup path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from rich.text import Text
from textual.widgets import DataTable, Static

from dbsprout.state.db import StateDB

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from textual.app import ComposeResult

    from dbsprout.state.models import RunRecord

logger = logging.getLogger(__name__)

#: Default state-DB location, matching :class:`~dbsprout.state.db.StateDB`.
_DEFAULT_DB_PATH = ".dbsprout/state.db"

#: Fidelity scores at or above this are healthy; below -> warn. Mirrors
#: ``dbsprout.report.quality_table._FIDELITY_WARN_THRESHOLD``.
_FIDELITY_WARN_THRESHOLD = 0.8

#: DataTable column headers, in display order.
_COLUMNS: tuple[str, ...] = ("Metric Type", "Metric Name", "Score", "Status", "Details")

Status = Literal["pass", "fail", "warn"]

_STATUS_COLORS: dict[Status, str] = {  # nosec B105 - "pass" is a status name, not a secret
    "pass": "green",
    "fail": "red",
    "warn": "yellow",
}

_EMPTY_MESSAGE = (
    "No quality results yet — run [b]dbsprout generate[/b] "
    "(or [b]dbsprout validate[/b]) to populate the quality table."
)


# ── Pure helpers: classification / colour ───────────────────────────────────


def quality_status(metric_type: str, score: float, *, passed: bool) -> Status:
    """Classify a quality result into ``pass`` / ``fail`` / ``warn``.

    Mirrors the HTML report's classifier: a check that did not pass is
    ``fail``; a *fidelity* metric that passed but scores below the healthy
    threshold is ``warn``; everything else is ``pass``. Integrity rows
    (``score=0.0``) classify as ``pass`` and never spuriously ``warn``.
    """
    if not passed:
        return "fail"
    if metric_type == "fidelity" and score < _FIDELITY_WARN_THRESHOLD:
        return "warn"
    return "pass"


def _status_color(status: Status) -> str:
    """Rich/Textual colour name used to style a status cell."""
    return _STATUS_COLORS[status]


def _status_cell(status: Status) -> Text:
    """A Rich ``Text`` status label (e.g. ``PASS``) styled with its colour."""
    return Text(status.upper(), style=_status_color(status))


# ── Pure helpers: formatting ─────────────────────────────────────────────────


def _format_score(metric_type: str, score: float) -> str:
    """Format a metric score for display.

    Integrity checks carry no meaningful numeric score (always ``0.0``), so
    they render as a dash; scored metrics (fidelity/detection) render to two
    decimal places.
    """
    if metric_type == "integrity":
        return "--"
    return f"{score:.2f}"


def _format_details(details_json: str | None) -> str:
    """Surface a check's detail string (empty when absent)."""
    return details_json or ""


# ── Pure helper: row reshaping ───────────────────────────────────────────────


@dataclass(frozen=True)
class QualityRow:
    """Immutable per-check row for the Quality table."""

    metric_type: str
    metric_name: str
    score: float
    status: Status
    details: str

    def cells(self) -> tuple[str, str, str, Text, str]:
        """Render this row to the DataTable cell tuple (status styled)."""
        return (
            self.metric_type,
            self.metric_name,
            _format_score(self.metric_type, self.score),
            _status_cell(self.status),
            self.details,
        )


def quality_rows(run: RunRecord | None) -> tuple[QualityRow, ...]:
    """Reshape a run's quality results into ordered :class:`QualityRow`\\ s.

    Returns an empty tuple for ``None`` (no runs recorded) or a run with no
    quality results, so the widget can render its friendly empty state.
    """
    if run is None:
        return ()
    return tuple(
        QualityRow(
            metric_type=q.metric_type,
            metric_name=q.metric_name,
            score=q.score,
            status=quality_status(q.metric_type, q.score, passed=q.passed),
            details=_format_details(q.details_json),
        )
        for q in run.quality_results
    )


def _empty_message() -> str:
    """The friendly placeholder shown when there are no quality results."""
    return _EMPTY_MESSAGE


def _summary_text(rows: Sequence[QualityRow]) -> str:
    """Summary line for the loaded rows (or the empty placeholder)."""
    if not rows:
        return _empty_message()
    passed = sum(1 for r in rows if r.status == "pass")
    warned = sum(1 for r in rows if r.status == "warn")
    failed = sum(1 for r in rows if r.status == "fail")
    return (
        f"Checks: {len(rows)}  |  "
        f"[green]Pass: {passed}[/green]  |  "
        f"[yellow]Warn: {warned}[/yellow]  |  "
        f"[red]Fail: {failed}[/red]"
    )


# ── Widget ────────────────────────────────────────────────────────────────────


class QualityScreen(Static):
    """Textual widget rendering the latest run's quality-check results."""

    DEFAULT_CSS: ClassVar[str] = """
    QualityScreen {
        height: 1fr;
        width: 1fr;
        layout: vertical;
    }
    QualityScreen #quality-summary {
        height: auto;
        width: 1fr;
        padding: 0 0 1 0;
    }
    QualityScreen #quality-table {
        height: 1fr;
        width: 1fr;
    }
    """

    def __init__(
        self,
        *,
        db_path: Path | str = _DEFAULT_DB_PATH,
        runs_loader: Callable[[], list[RunRecord]] | None = None,
    ) -> None:
        """Build the screen.

        *runs_loader* is an injection seam for tests: when ``None`` the screen
        reads the newest run from the on-disk state DB at *db_path*; tests pass
        a callable returning in-memory runs to avoid touching disk.
        """
        super().__init__()
        self._db_path = Path(db_path)
        self._runs_loader = runs_loader

    def compose(self) -> ComposeResult:
        yield Static(_empty_message(), id="quality-summary")
        yield DataTable(id="quality-table")

    def on_mount(self) -> None:
        """Set up columns then render the newest run's results."""
        table = self.query_one("#quality-table", DataTable)
        table.add_columns(*_COLUMNS)
        table.cursor_type = "row"
        table.zebra_stripes = True
        self._render_rows(quality_rows(self._latest_run()))

    def _latest_run(self) -> RunRecord | None:
        """Read the newest run from the state DB (or injected loader); never raise."""
        try:
            runs = self._runs_loader() if self._runs_loader is not None else self._load_runs()
        except Exception as exc:  # pragma: no cover - defensive; telemetry is optional
            logger.warning("Could not read state DB for TUI quality (%s); showing empty.", exc)
            return None
        return runs[0] if runs else None

    def _load_runs(self) -> list[RunRecord]:
        return StateDB(self._db_path).get_runs()

    def _render_rows(self, rows: Sequence[QualityRow]) -> None:
        """Refresh the summary line and (re)populate the table rows."""
        self.query_one("#quality-summary", Static).update(_summary_text(rows))
        table = self.query_one("#quality-table", DataTable)
        for row in rows:
            table.add_row(*row.cells())
