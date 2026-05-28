"""TUI Progress screen (S-087).

Renders real-time generation progress in the Progress tab of
:class:`~dbsprout.tui.app.DBSproutApp`:

* an overall summary line (total rows / total tables / elapsed),
* an overall completion :class:`~textual.widgets.ProgressBar` with ETA, and
* a per-table :class:`~textual.widgets.DataTable` (status / rows / rows-per-sec
  / ETA) with colour-coded status cells.

**Data source.** Read-only polling of the shared state DB
(``.dbsprout/state.db``) via :class:`~dbsprout.state.db.StateDB`. DBSprout
generation is a CLI flow that writes per-table telemetry once a table
finishes; the TUI is a *visual surface* that reads the newest run every
``poll_interval`` seconds. A table that is not yet present in the latest run
is **pending**; one that is present is **complete** (or **error** when its
``errors`` count is non-zero). The ``generating`` (blue) status is wired
through the public :meth:`ProgressScreen.update_snapshot` API for a future
streaming writer; the poll path resolves the honest pending/complete/error
signal the state layer supplies today rather than fabricating intra-table
percentages.

**Flicker-free updates.** A single :class:`DataTable` is mounted once and its
cells are updated *in place* (``update_cell``) on each poll rather than being
cleared and rebuilt; the summary and overall bar are driven by reactive
attributes.

``textual`` is the optional ``[tui]`` extra. This module is imported lazily
(only by the ``dbsprout tui`` command), so the top-level textual import never
touches the hot CLI-startup path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import DataTable, ProgressBar, Static

from dbsprout.state.db import StateDB

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.widgets.data_table import ColumnKey

    from dbsprout.state.models import RunRecord

logger = logging.getLogger(__name__)

#: Default state-DB location, matching :class:`~dbsprout.state.db.StateDB`.
_DEFAULT_DB_PATH = ".dbsprout/state.db"

#: How often (seconds) the screen re-reads the state DB.
_DEFAULT_POLL_INTERVAL = 0.5

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600

_COLUMNS: tuple[str, ...] = ("Table", "Status", "Rows", "Rows/sec", "ETA")


class TableStatus(str, Enum):
    """Lifecycle status of a single table, with a render colour."""

    PENDING = "pending"
    GENERATING = "generating"
    COMPLETE = "complete"
    ERROR = "error"

    @property
    def color(self) -> str:
        """Rich/Textual colour name used to style the status cell."""
        return _STATUS_COLORS[self]

    @property
    def label(self) -> str:
        """Human-readable, title-cased status label."""
        return self.value.title()


_STATUS_COLORS: dict[TableStatus, str] = {
    TableStatus.PENDING: "gray",
    TableStatus.GENERATING: "blue",
    TableStatus.COMPLETE: "green",
    TableStatus.ERROR: "red",
}


def format_eta(seconds: float | None) -> str:
    """Format an ETA in seconds as a compact human string.

    ``None`` (unknown) renders as ``"--"``; otherwise the largest two units
    are shown (``"5s"``, ``"1m 5s"``, ``"1h 2m"``).
    """
    if seconds is None:
        return "--"
    total = max(0, int(seconds))
    if total < _SECONDS_PER_MINUTE:
        return f"{total}s"
    if total < _SECONDS_PER_HOUR:
        minutes, secs = divmod(total, _SECONDS_PER_MINUTE)
        return f"{minutes}m {secs}s"
    hours, remainder = divmod(total, _SECONDS_PER_HOUR)
    minutes = remainder // _SECONDS_PER_MINUTE
    return f"{hours}h {minutes}m"


@dataclass(frozen=True)
class TableProgress:
    """Immutable per-table progress row for the screen."""

    name: str
    status: TableStatus
    row_count: int = 0
    rows_per_sec: float = 0.0
    eta_seconds: float | None = None


@dataclass(frozen=True)
class ProgressSnapshot:
    """Immutable point-in-time view of the newest generation run.

    ``is_empty`` distinguishes "no runs recorded at all" (placeholder) from a
    run that simply has no tables yet.
    """

    total_rows: int = 0
    total_tables: int = 0
    recorded_tables: int = 0
    elapsed_seconds: float = 0.0
    overall_rows_per_sec: float = 0.0
    overall_eta_seconds: float | None = None
    tables: tuple[TableProgress, ...] = ()
    is_empty: bool = True

    @classmethod
    def empty(cls) -> ProgressSnapshot:
        """Return the canonical "no runs yet" snapshot."""
        return cls()

    @property
    def completion_fraction(self) -> float:
        """Fraction (0.0-1.0) of tables recorded for the latest run."""
        if self.total_tables <= 0:
            return 0.0
        return min(1.0, self.recorded_tables / self.total_tables)


def _elapsed_seconds(run: RunRecord, now: datetime) -> float:
    """Elapsed wall time for a run.

    A completed run prefers its recorded ``duration_ms``; an in-flight run
    uses ``now - started_at``.
    """
    if run.completed_at is not None and run.duration_ms is not None:
        return run.duration_ms / 1000.0
    started = run.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    reference = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    return max(0.0, (reference - started).total_seconds())


def _overall_rates(
    *, recorded_rows: int, total_rows: int, elapsed_seconds: float, all_recorded: bool
) -> tuple[float, float | None]:
    """Compute overall rows/sec and ETA seconds.

    ETA is ``0.0`` when every table is recorded (no remaining work) and
    ``None`` (unknown) when no throughput is yet observable.
    """
    rows_per_sec = recorded_rows / elapsed_seconds if elapsed_seconds > 0 else 0.0
    if all_recorded:
        return rows_per_sec, 0.0
    remaining = max(0, total_rows - recorded_rows)
    if rows_per_sec <= 0:
        return rows_per_sec, None
    return rows_per_sec, remaining / rows_per_sec


def snapshot_from_run(run: RunRecord | None, *, now: datetime | None = None) -> ProgressSnapshot:
    """Build a :class:`ProgressSnapshot` from the newest run (or ``None``)."""
    if run is None:
        return ProgressSnapshot.empty()

    reference = now or datetime.now(tz=timezone.utc)
    elapsed = _elapsed_seconds(run, reference)
    recorded_rows = sum(stat.row_count for stat in run.table_stats)
    recorded_tables = len(run.table_stats)
    all_recorded = recorded_tables >= run.total_tables

    tables: list[TableProgress] = [
        TableProgress(
            name=stat.table_name,
            status=TableStatus.ERROR if stat.errors > 0 else TableStatus.COMPLETE,
            row_count=stat.row_count,
            rows_per_sec=stat.rows_per_sec,
            eta_seconds=0.0,
        )
        for stat in run.table_stats
    ]
    # Synthesise pending placeholder rows for tables not yet recorded.
    for index in range(recorded_tables, run.total_tables):
        tables.append(TableProgress(name=f"(pending #{index + 1})", status=TableStatus.PENDING))

    rows_per_sec, eta = _overall_rates(
        recorded_rows=recorded_rows,
        total_rows=run.total_rows,
        elapsed_seconds=elapsed,
        all_recorded=all_recorded,
    )
    return ProgressSnapshot(
        total_rows=run.total_rows,
        total_tables=run.total_tables,
        recorded_tables=recorded_tables,
        elapsed_seconds=elapsed,
        overall_rows_per_sec=rows_per_sec,
        overall_eta_seconds=eta,
        tables=tuple(tables),
        is_empty=False,
    )


def _status_cell(status: TableStatus) -> Text:
    """A Rich ``Text`` status label styled with the status colour."""
    return Text(status.label, style=status.color)


def _summary_text(snapshot: ProgressSnapshot) -> str:
    """The top summary line for a snapshot."""
    if snapshot.is_empty:
        return "No generation runs yet — run `dbsprout generate` to populate progress."
    return (
        f"Rows: {snapshot.total_rows:,}  |  "
        f"Tables: {snapshot.recorded_tables}/{snapshot.total_tables}  |  "
        f"Elapsed: {format_eta(snapshot.elapsed_seconds)}  |  "
        f"Rows/sec: {snapshot.overall_rows_per_sec:,.0f}  |  "
        f"ETA: {format_eta(snapshot.overall_eta_seconds)}"
    )


@dataclass
class _RowState:
    """Tracks DataTable row keys so updates happen in place (flicker-free)."""

    keys: dict[str, str] = field(default_factory=dict)


class ProgressScreen(Static):
    """Textual widget rendering the live generation-progress dashboard."""

    DEFAULT_CSS: ClassVar[str] = """
    ProgressScreen {
        height: 1fr;
        width: 1fr;
        layout: vertical;
    }
    ProgressScreen #progress-summary {
        height: auto;
        width: 1fr;
        padding: 0 0 1 0;
    }
    ProgressScreen #progress-overall {
        width: 1fr;
        height: auto;
        padding: 0 0 1 0;
    }
    ProgressScreen #progress-table {
        height: 1fr;
        width: 1fr;
    }
    """

    snapshot: reactive[ProgressSnapshot] = reactive(
        ProgressSnapshot.empty, init=False, always_update=True
    )

    def __init__(
        self,
        *,
        db_path: Path | str = _DEFAULT_DB_PATH,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
    ) -> None:
        super().__init__()
        self._db_path = Path(db_path)
        self._poll_interval = poll_interval
        self._row_state = _RowState()
        self._columns_ready = False
        self._column_keys: list[ColumnKey] = []

    def compose(self) -> ComposeResult:
        yield Static(_summary_text(ProgressSnapshot.empty()), id="progress-summary")
        yield ProgressBar(total=100.0, show_eta=False, id="progress-overall")
        yield DataTable(id="progress-table")

    def on_mount(self) -> None:
        """Set up columns, do an immediate poll, then poll periodically.

        Mount setup is made idempotent. On Textual 8.x the mount handler can
        fire more than once for the same widget; without a guard the second
        pass re-adds the columns and re-runs the initial render, so a snapshot
        that should produce N rows ends up with 2*N rows (the duplicate-row
        regression: a two-table snapshot rendered 4 rows instead of 2). The
        ``_columns_ready`` guard ensures column setup and the initial render
        happen exactly once, so later snapshots update rows in place.
        """
        if self._columns_ready:
            return
        table = self.query_one("#progress-table", DataTable)
        self._column_keys = table.add_columns(*_COLUMNS)
        table.cursor_type = "row"
        table.zebra_stripes = True
        self._columns_ready = True
        self._render_snapshot(self.snapshot)
        self._poll()
        self.set_interval(self._poll_interval, self._poll)

    def _poll(self) -> None:
        """Read the newest run from the state DB and refresh; never raise."""
        try:
            runs = StateDB(self._db_path).get_runs()
        except Exception as exc:  # pragma: no cover - defensive; telemetry is optional
            logger.warning("Could not read state DB for TUI progress (%s); keeping last view.", exc)
            return
        self.update_from_run(runs[0] if runs else None)

    def update_from_run(self, run: RunRecord | None) -> None:
        """Recompute the snapshot from a run and refresh the UI."""
        self.update_snapshot(snapshot_from_run(run))

    def update_snapshot(self, snapshot: ProgressSnapshot) -> None:
        """Set the reactive snapshot (triggers ``watch_snapshot``)."""
        self.snapshot = snapshot

    def watch_snapshot(self, snapshot: ProgressSnapshot) -> None:
        """React to a new snapshot: refresh summary, bar, and table rows."""
        self._render_snapshot(snapshot)

    def _render_snapshot(self, snapshot: ProgressSnapshot) -> None:
        """Refresh summary, overall bar, and table rows once children exist."""
        if not self._columns_ready:
            return
        self.query_one("#progress-summary", Static).update(_summary_text(snapshot))
        bar = self.query_one("#progress-overall", ProgressBar)
        bar.update(total=100.0, progress=snapshot.completion_fraction * 100.0)
        self._refresh_rows(snapshot)

    def _refresh_rows(self, snapshot: ProgressSnapshot) -> None:
        """Add/update table rows in place to avoid clear-and-rebuild flicker."""
        table = self.query_one("#progress-table", DataTable)
        for entry in snapshot.tables:
            cells = (
                entry.name,
                _status_cell(entry.status),
                f"{entry.row_count:,}",
                f"{entry.rows_per_sec:,.0f}",
                format_eta(entry.eta_seconds),
            )
            key = self._row_state.keys.get(entry.name)
            if key is None:
                self._row_state.keys[entry.name] = entry.name
                table.add_row(*cells, key=entry.name)
            else:
                for column_key, value in zip(self._column_keys, cells, strict=True):
                    table.update_cell(key, column_key, value)
