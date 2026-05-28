"""TUI quality-display tests (S-089).

Textual is an optional ``[tui]`` extra; the whole module is skipped when it is
absent (mirrors ``tests/test_tui/test_app.py`` and ``test_schema.py``). Async
widget scenarios wrap :meth:`textual.app.App.run_test` in ``asyncio.run(...)``
inside synchronous test functions — the repo has no async pytest plugin.

Pure helpers (status classification, colour map, row reshaping, formatters,
empty-state) are exercised directly without a running app for fast,
deterministic coverage.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("textual", reason="textual absent (pip install dbsprout[tui])")

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

from dbsprout.state.models import QualityResult, RunRecord
from dbsprout.tui.app import DBSproutApp
from dbsprout.tui.screens.quality import (
    _COLUMNS,
    QualityRow,
    QualityScreen,
    _empty_message,
    _format_details,
    _format_score,
    _status_cell,
    _status_color,
    quality_rows,
    quality_status,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from typing import TypeVar

    _T = TypeVar("_T")


# ── Fixtures ────────────────────────────────────────────────────────────────


def _run(coro: Awaitable[_T]) -> _T:
    """Run an async pilot coroutine in a fresh event loop."""
    return asyncio.run(coro)  # type: ignore[arg-type]


def _quality_result(
    *,
    metric_type: str = "integrity",
    metric_name: str = "fk_integrity",
    score: float = 0.0,
    passed: bool = True,
    details_json: str | None = None,
) -> QualityResult:
    return QualityResult(
        metric_type=metric_type,
        metric_name=metric_name,
        score=score,
        passed=passed,
        details_json=details_json,
    )


def _run_record(*results: QualityResult) -> RunRecord:
    return RunRecord(
        started_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
        engine="heuristic",
        quality_results=list(results),
    )


class _QualityHarness(App[None]):
    """Minimal host that mounts a ``QualityScreen`` with injected runs.

    Mounting the widget directly (rather than through ``DBSproutApp``) keeps
    rendering tests deterministic: a fake ``runs_loader`` replaces the on-disk
    state-DB read so tests never touch ``.dbsprout/state.db``.
    """

    def __init__(self, runs: list[RunRecord]) -> None:
        super().__init__()
        self._runs = runs

    def compose(self) -> ComposeResult:
        yield QualityScreen(runs_loader=lambda: self._runs)


# ── Pure helper: status classification ──────────────────────────────────────


def test_status_fail_when_not_passed() -> None:
    assert quality_status("integrity", 0.0, passed=False) == "fail"
    assert quality_status("fidelity", 0.95, passed=False) == "fail"


def test_status_pass_for_passing_integrity_check() -> None:
    assert quality_status("integrity", 0.0, passed=True) == "pass"


def test_status_warn_for_low_fidelity_score() -> None:
    assert quality_status("fidelity", 0.5, passed=True) == "warn"


def test_status_pass_for_healthy_fidelity_score() -> None:
    assert quality_status("fidelity", 0.95, passed=True) == "pass"


def test_status_does_not_warn_on_non_fidelity_zero_score() -> None:
    # Integrity rows carry score 0.0 but must classify as pass, never warn.
    assert quality_status("integrity", 0.0, passed=True) == "pass"
    assert quality_status("detection", 0.0, passed=True) == "pass"


# ── Pure helper: status colour + cell ────────────────────────────────────────


def test_status_color_map() -> None:
    assert _status_color("pass") == "green"
    assert _status_color("fail") == "red"
    assert _status_color("warn") == "yellow"


def test_status_cell_is_styled_text() -> None:
    cell = _status_cell("pass")
    assert isinstance(cell, Text)
    assert cell.style == "green"
    assert "PASS" in str(cell)

    fail_cell = _status_cell("fail")
    assert fail_cell.style == "red"
    assert "FAIL" in str(fail_cell)


# ── Pure helper: row reshaping ───────────────────────────────────────────────


def test_quality_rows_maps_results_in_order() -> None:
    run = _run_record(
        _quality_result(metric_name="fk_integrity", passed=True),
        _quality_result(metric_name="not_null", passed=False),
    )
    rows = quality_rows(run)
    assert [r.metric_name for r in rows] == ["fk_integrity", "not_null"]
    assert isinstance(rows[0], QualityRow)
    assert rows[0].status == "pass"
    assert rows[1].status == "fail"


def test_quality_rows_none_run_is_empty() -> None:
    assert quality_rows(None) == ()


def test_quality_rows_run_with_no_results_is_empty() -> None:
    assert quality_rows(_run_record()) == ()


def test_quality_rows_carries_fidelity_status() -> None:
    run = _run_record(
        _quality_result(metric_type="fidelity", metric_name="ks_stat", score=0.5, passed=True),
    )
    rows = quality_rows(run)
    assert rows[0].status == "warn"
    assert rows[0].metric_type == "fidelity"


# ── Pure helper: formatters ──────────────────────────────────────────────────


def test_format_score_blank_for_zero_integrity() -> None:
    # Integrity checks carry no meaningful numeric score (0.0) -> dash.
    assert _format_score("integrity", 0.0) == "--"


def test_format_score_formats_fidelity() -> None:
    assert _format_score("fidelity", 0.875) == "0.88"


def test_format_details_passthrough_and_empty() -> None:
    assert _format_details(None) == ""
    assert "3 violations" in _format_details("3 violations")


# ── Pure helper: empty state ─────────────────────────────────────────────────


def test_empty_message_mentions_generate() -> None:
    msg = _empty_message()
    assert "generate" in msg.lower()


# ── Widget: table render ─────────────────────────────────────────────────────


def test_widget_renders_one_row_per_result() -> None:
    async def _scenario() -> None:
        run = _run_record(
            _quality_result(metric_name="fk_integrity", passed=True),
            _quality_result(metric_name="not_null", passed=False, details_json="2 nulls"),
        )
        app = _QualityHarness([run])
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = pilot.app.query_one(QualityScreen)
            table = screen.query_one(DataTable)
            assert table.row_count == 2
            header_labels = [str(col.label) for col in table.columns.values()]
            assert list(_COLUMNS) == header_labels

    _run(_scenario())


def test_widget_newest_run_used_by_default() -> None:
    async def _scenario() -> None:
        old = _run_record(_quality_result(metric_name="old_check"))
        newest = _run_record(
            _quality_result(metric_name="new_check_a"),
            _quality_result(metric_name="new_check_b"),
        )
        # get_runs() returns newest-first; harness loader mirrors that order.
        app = _QualityHarness([newest, old])
        async with app.run_test() as pilot:
            await pilot.pause()
            table = pilot.app.query_one(QualityScreen).query_one(DataTable)
            assert table.row_count == 2

    _run(_scenario())


def test_widget_pass_fail_colour() -> None:
    async def _scenario() -> None:
        run = _run_record(
            _quality_result(metric_name="ok_check", passed=True),
            _quality_result(metric_name="bad_check", passed=False),
        )
        app = _QualityHarness([run])
        async with app.run_test() as pilot:
            await pilot.pause()
            table = pilot.app.query_one(QualityScreen).query_one(DataTable)
            status_styles = []
            for row_key in table.rows:
                cell = table.get_row(row_key)[_COLUMNS.index("Status")]
                assert isinstance(cell, Text)
                status_styles.append(str(cell.style))
            assert "green" in status_styles
            assert "red" in status_styles

    _run(_scenario())


def test_widget_empty_state_message() -> None:
    async def _scenario() -> None:
        app = _QualityHarness([])
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = pilot.app.query_one(QualityScreen)
            summary = screen.query_one("#quality-summary", Static)
            rendered = str(summary.render())
            assert "generate" in rendered.lower()
            table = screen.query_one(DataTable)
            assert table.row_count == 0

    _run(_scenario())


def test_widget_load_failure_is_swallowed() -> None:
    async def _scenario() -> None:
        def _boom() -> list[RunRecord]:
            raise RuntimeError("state db unreadable")

        class _BoomHarness(App[None]):
            def compose(self) -> ComposeResult:
                yield QualityScreen(runs_loader=_boom)

        app = _BoomHarness()
        async with app.run_test() as pilot:
            await pilot.pause()
            # A read failure must degrade to the empty state, never crash.
            screen = pilot.app.query_one(QualityScreen)
            table = screen.query_one(DataTable)
            assert table.row_count == 0

    _run(_scenario())


# ── App wiring ───────────────────────────────────────────────────────────────


def test_quality_tab_hosts_quality_screen() -> None:
    async def _scenario() -> None:
        app = DBSproutApp()
        async with app.run_test() as pilot:
            await pilot.press("u")  # switch to Quality tab
            await pilot.pause()
            assert pilot.app.query(QualityScreen)

    _run(_scenario())


def test_settings_tab_remains_placeholder() -> None:
    async def _scenario() -> None:
        app = DBSproutApp()
        async with app.run_test() as pilot:
            await pilot.press("g")  # Settings tab
            await pilot.pause()
            settings_pane = pilot.app.query_one("#settings")
            assert settings_pane.query(Static)
            assert not settings_pane.query(QualityScreen)

    _run(_scenario())
