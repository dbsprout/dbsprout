"""TUI progress-screen tests (S-087).

Two layers are tested:

1. A textual-free *model* layer (``snapshot_from_run``, ``TableStatus``,
   ``format_eta`` …) that turns the newest state-DB :class:`RunRecord` into a
   plain :class:`ProgressSnapshot` — unit-testable without a terminal.
2. The :class:`ProgressScreen` Textual widget, driven headlessly via
   ``app.run_test()`` wrapped in ``asyncio.run(...)`` inside synchronous test
   functions (the repo has no async pytest plugin), mirroring
   ``tests/test_tui/test_app.py``.

``textual`` is the optional ``[tui]`` extra, so this module skips wholesale
when it is absent (the importorskip guard runs BEFORE any textual import).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("textual", reason="textual absent (pip install dbsprout[tui])")

from textual.app import App, ComposeResult
from textual.widgets import DataTable, ProgressBar, Static

from dbsprout.state.models import RunRecord, TableStats
from dbsprout.tui.screens.progress import (
    ProgressScreen,
    ProgressSnapshot,
    TableStatus,
    format_eta,
    snapshot_from_run,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from pathlib import Path
    from typing import TypeVar

    _T = TypeVar("_T")


def _run(coro: Awaitable[_T]) -> _T:
    """Run an async pilot coroutine in a fresh event loop."""
    return asyncio.run(coro)  # type: ignore[arg-type]


_NOW = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)


def _run_record(  # noqa: PLR0913
    *,
    started_at: datetime,
    completed_at: datetime | None,
    total_rows: int,
    total_tables: int,
    table_stats: list[TableStats],
    duration_ms: int | None = None,
) -> RunRecord:
    return RunRecord(
        id=1,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
        engine="heuristic",
        total_rows=total_rows,
        total_tables=total_tables,
        table_stats=table_stats,
    )


# ── model layer ─────────────────────────────────────────────────────────


def test_table_status_color_map() -> None:
    assert TableStatus.PENDING.color == "gray"
    assert TableStatus.GENERATING.color == "blue"
    assert TableStatus.COMPLETE.color == "green"
    assert TableStatus.ERROR.color == "red"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, "--"),
        (0.0, "0s"),
        (5.4, "5s"),
        (65.0, "1m 5s"),
        (3725.0, "1h 2m"),
    ],
)
def test_format_eta(seconds: float | None, expected: str) -> None:
    assert format_eta(seconds) == expected


def test_snapshot_from_none_is_empty() -> None:
    snap = snapshot_from_run(None, now=_NOW)
    assert snap.is_empty
    assert snap.total_rows == 0
    assert snap.total_tables == 0
    assert snap.tables == ()
    assert snap.completion_fraction == 0.0


def test_snapshot_from_empty_run_has_zero_tables() -> None:
    run = _run_record(
        started_at=_NOW,
        completed_at=_NOW,
        total_rows=0,
        total_tables=0,
        table_stats=[],
        duration_ms=0,
    )
    snap = snapshot_from_run(run, now=_NOW)
    assert not snap.is_empty  # a run exists, it just has no tables yet
    assert snap.total_tables == 0
    assert snap.tables == ()
    assert snap.completion_fraction == 0.0


def test_snapshot_marks_complete_and_error_tables() -> None:
    run = _run_record(
        started_at=_NOW - timedelta(seconds=10),
        completed_at=_NOW,
        total_rows=300,
        total_tables=3,
        duration_ms=10_000,
        table_stats=[
            TableStats(table_name="users", row_count=100, generation_ms=1000, rows_per_sec=100.0),
            TableStats(table_name="orders", row_count=200, generation_ms=4000, rows_per_sec=50.0),
            TableStats(table_name="bad", row_count=0, generation_ms=0, rows_per_sec=0.0, errors=2),
        ],
    )
    snap = snapshot_from_run(run, now=_NOW)
    by_name = {t.name: t for t in snap.tables}
    assert by_name["users"].status is TableStatus.COMPLETE
    assert by_name["orders"].status is TableStatus.COMPLETE
    assert by_name["bad"].status is TableStatus.ERROR
    assert snap.recorded_tables == 3
    assert snap.completion_fraction == pytest.approx(1.0)


def test_snapshot_pending_tables_when_run_in_flight() -> None:
    # total_tables=4 but only 1 recorded yet, and the run has not completed.
    run = _run_record(
        started_at=_NOW - timedelta(seconds=5),
        completed_at=None,
        total_rows=100,
        total_tables=4,
        table_stats=[
            TableStats(table_name="users", row_count=100, generation_ms=1000, rows_per_sec=100.0),
        ],
    )
    snap = snapshot_from_run(run, now=_NOW)
    statuses = sorted(t.status for t in snap.tables)
    # 1 complete + 3 pending = 4 rows total.
    assert len(snap.tables) == 4
    assert statuses.count(TableStatus.COMPLETE) == 1
    assert statuses.count(TableStatus.PENDING) == 3
    # Elapsed uses now - started_at for an in-flight run.
    assert snap.elapsed_seconds == pytest.approx(5.0)
    assert snap.completion_fraction == pytest.approx(0.25)


def test_snapshot_overall_rows_per_sec_and_eta() -> None:
    # 100 rows recorded in 5s elapsed => 20 rows/sec; 300 more remaining => 15s ETA.
    run = _run_record(
        started_at=_NOW - timedelta(seconds=5),
        completed_at=None,
        total_rows=400,
        total_tables=2,
        table_stats=[
            TableStats(table_name="users", row_count=100, generation_ms=5000, rows_per_sec=20.0),
        ],
    )
    snap = snapshot_from_run(run, now=_NOW)
    assert snap.overall_rows_per_sec == pytest.approx(20.0)
    assert snap.overall_eta_seconds == pytest.approx(15.0)


def test_snapshot_completed_run_uses_duration_ms_for_elapsed() -> None:
    run = _run_record(
        started_at=_NOW - timedelta(seconds=99),
        completed_at=_NOW,
        total_rows=10,
        total_tables=1,
        duration_ms=8000,
        table_stats=[
            TableStats(table_name="users", row_count=10, generation_ms=8000, rows_per_sec=1.25),
        ],
    )
    snap = snapshot_from_run(run, now=_NOW)
    # Completed run prefers duration_ms over wall-clock now-started.
    assert snap.elapsed_seconds == pytest.approx(8.0)
    # All tables done => no remaining work => ETA 0.
    assert snap.overall_eta_seconds == pytest.approx(0.0)


def test_snapshot_naive_started_at_is_treated_as_utc() -> None:
    # A run whose started_at is naive (no tzinfo) must not raise when
    # subtracted from an aware ``now``; it is treated as UTC.
    naive_start = datetime(2026, 5, 20, 11, 59, 50)  # intentional naive value (treated as UTC)
    run = _run_record(
        started_at=naive_start,
        completed_at=None,
        total_rows=100,
        total_tables=2,
        table_stats=[
            TableStats(table_name="users", row_count=50, generation_ms=5000, rows_per_sec=10.0),
        ],
    )
    snap = snapshot_from_run(run, now=_NOW)
    assert snap.elapsed_seconds == pytest.approx(10.0)


def test_snapshot_eta_unknown_when_no_throughput_yet() -> None:
    # In-flight run, no elapsed time observed yet => overall rows/sec 0 =>
    # ETA is unknown (None) rather than infinite/zero.
    run = _run_record(
        started_at=_NOW,
        completed_at=None,
        total_rows=100,
        total_tables=2,
        table_stats=[
            TableStats(table_name="users", row_count=0, generation_ms=0, rows_per_sec=0.0),
        ],
    )
    snap = snapshot_from_run(run, now=_NOW)
    assert snap.overall_rows_per_sec == pytest.approx(0.0)
    assert snap.overall_eta_seconds is None


# ── widget layer ────────────────────────────────────────────────────────


class _Harness(App[None]):
    """Minimal host app mounting a single :class:`ProgressScreen`."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        super().__init__()
        self._db_path = db_path

    def compose(self) -> ComposeResult:
        if self._db_path is None:
            yield ProgressScreen()
        else:
            yield ProgressScreen(db_path=self._db_path, poll_interval=1000.0)


def test_update_snapshot_before_mount_is_a_noop() -> None:
    # Calling update_snapshot before the widget mounts (columns not ready)
    # must not raise — the render is deferred until on_mount.
    screen = ProgressScreen()
    screen.update_snapshot(ProgressSnapshot.empty())
    assert screen.snapshot.is_empty


def test_mount_is_idempotent_and_does_not_duplicate_rows() -> None:
    # Root cause of the duplicate-row failures: on Textual 8.x the mount
    # handler can fire more than once. Without a guard, the second pass
    # re-adds the columns and re-runs the initial render, so a two-table
    # snapshot is rendered into 4 rows instead of 2 (the assert 4 == 2 /
    # assert 2 == 1 regression). Mount setup must be idempotent: exactly five
    # columns, and rendering a snapshot must yield one row per table no matter
    # how many times the mount/setup path fires.
    snap = snapshot_from_run(
        _run_record(
            started_at=_NOW - timedelta(seconds=10),
            completed_at=_NOW,
            total_rows=300,
            total_tables=2,
            duration_ms=10_000,
            table_stats=[
                TableStats(
                    table_name="users", row_count=100, generation_ms=1000, rows_per_sec=100.0
                ),
                TableStats(
                    table_name="orders", row_count=200, generation_ms=4000, rows_per_sec=50.0
                ),
            ],
        ),
        now=_NOW,
    )

    async def _scenario() -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            screen = pilot.app.query_one(ProgressScreen)
            # Simulate a second mount-handler firing (the version-delta bug).
            screen.on_mount()
            await pilot.pause()
            screen.update_snapshot(snap)
            await pilot.pause()
            table = screen.query_one(DataTable)
            assert len(table.columns) == 5
            assert len(screen._column_keys) == 5
            assert table.row_count == 2  # one row per table, no duplicates

    _run(_scenario())


def test_widget_renders_one_row_per_table_with_status_color() -> None:
    snap = snapshot_from_run(
        _run_record(
            started_at=_NOW - timedelta(seconds=10),
            completed_at=_NOW,
            total_rows=300,
            total_tables=2,
            duration_ms=10_000,
            table_stats=[
                TableStats(
                    table_name="users", row_count=100, generation_ms=1000, rows_per_sec=100.0
                ),
                TableStats(
                    table_name="bad", row_count=0, generation_ms=0, rows_per_sec=0.0, errors=1
                ),
            ],
        ),
        now=_NOW,
    )

    async def _scenario() -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            screen = pilot.app.query_one(ProgressScreen)
            screen.update_snapshot(snap)
            await pilot.pause()
            table = screen.query_one(DataTable)
            assert table.row_count == 2
            # Status cells carry the per-status colour as a Rich Text style.
            statuses = {
                str(table.get_cell_at((r, 1))).strip().lower(): table.get_cell_at((r, 1))
                for r in range(table.row_count)
            }
            assert "complete" in statuses
            assert "error" in statuses
            complete_style = str(statuses["complete"].style)  # type: ignore[union-attr]
            error_style = str(statuses["error"].style)  # type: ignore[union-attr]
            assert "green" in complete_style
            assert "red" in error_style

    _run(_scenario())


def test_widget_summary_and_overall_bar() -> None:
    snap = snapshot_from_run(
        _run_record(
            started_at=_NOW - timedelta(seconds=5),
            completed_at=None,
            total_rows=400,
            total_tables=2,
            table_stats=[
                TableStats(
                    table_name="users", row_count=100, generation_ms=5000, rows_per_sec=20.0
                ),
            ],
        ),
        now=_NOW,
    )

    async def _scenario() -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            screen = pilot.app.query_one(ProgressScreen)
            screen.update_snapshot(snap)
            await pilot.pause()
            summary = screen.query_one("#progress-summary", Static)
            text = str(summary.render())
            assert "400" in text  # total rows
            assert "2" in text  # total tables
            bar = screen.query_one(ProgressBar)
            # Overall completion = 1 of 2 tables recorded => 50%.
            assert bar.progress == pytest.approx(50.0)

    _run(_scenario())


def test_widget_update_does_not_duplicate_rows() -> None:
    snap = snapshot_from_run(
        _run_record(
            started_at=_NOW - timedelta(seconds=10),
            completed_at=_NOW,
            total_rows=100,
            total_tables=1,
            duration_ms=10_000,
            table_stats=[
                TableStats(
                    table_name="users", row_count=100, generation_ms=1000, rows_per_sec=100.0
                ),
            ],
        ),
        now=_NOW,
    )

    async def _scenario() -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            screen = pilot.app.query_one(ProgressScreen)
            screen.update_snapshot(snap)
            await pilot.pause()
            screen.update_snapshot(snap)  # poll again with same data
            await pilot.pause()
            table = screen.query_one(DataTable)
            assert table.row_count == 1  # in-place update, no duplicate row

    _run(_scenario())


def test_widget_empty_state_shows_placeholder() -> None:
    async def _scenario() -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            screen = pilot.app.query_one(ProgressScreen)
            screen.update_snapshot(ProgressSnapshot.empty())
            await pilot.pause()
            summary = screen.query_one("#progress-summary", Static)
            assert "no generation runs" in str(summary.render()).lower()

    _run(_scenario())


def test_widget_polls_state_db(tmp_path: Path) -> None:
    from dbsprout.state.db import StateDB  # noqa: PLC0415

    db_path = tmp_path / "state.db"
    StateDB(db_path).record_run(
        _run_record(
            started_at=_NOW - timedelta(seconds=10),
            completed_at=_NOW,
            total_rows=100,
            total_tables=1,
            duration_ms=10_000,
            table_stats=[
                TableStats(
                    table_name="users", row_count=100, generation_ms=1000, rows_per_sec=100.0
                ),
            ],
        )
    )

    async def _scenario() -> None:
        app = _Harness(db_path=db_path)
        async with app.run_test() as pilot:
            screen = pilot.app.query_one(ProgressScreen)
            await pilot.pause()  # on_mount triggers an immediate poll
            table = screen.query_one(DataTable)
            assert table.row_count == 1
            assert str(table.get_cell_at((0, 0))) == "users"

    _run(_scenario())


def test_widget_missing_db_does_not_crash(tmp_path: Path) -> None:
    async def _scenario() -> None:
        app = _Harness(db_path=tmp_path / "absent.db")
        async with app.run_test() as pilot:
            screen = pilot.app.query_one(ProgressScreen)
            await pilot.pause()
            summary = screen.query_one("#progress-summary", Static)
            # StateDB creates an empty DB on open => zero runs => placeholder.
            assert "no generation runs" in str(summary.render()).lower()

    _run(_scenario())
