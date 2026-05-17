"""Self-contained HTML report generator (S-081).

Reads the newest generation run from the shared state DB (S-079/S-080) and
renders a single, dependency-free HTML file (all CSS/JS/data inline) that
opens in any browser with no server.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dbsprout.report.context import build_report_context
from dbsprout.report.env import render_report

if TYPE_CHECKING:
    from dbsprout.state import StateDB

#: Default location for the generated report.
DEFAULT_OUTPUT_PATH = Path("seeds") / "report.html"


class ReportGenerator:
    """Render a StateDB's newest run to a self-contained HTML report."""

    def __init__(self, output_path: Path | str | None = None) -> None:
        self._output_path = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_PATH

    @property
    def output_path(self) -> Path:
        """Resolved destination path for the rendered report."""
        return self._output_path

    def generate(self, state_db: StateDB) -> Path:
        """Render the newest run from ``state_db`` to an HTML file.

        Returns the path to the written report. The destination directory
        is created if missing. When the state DB has no runs, a graceful
        empty-state report is still produced.
        """
        runs = state_db.get_runs()
        newest = runs[0] if runs else None
        html = render_report(build_report_context(newest))

        out = self._output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return out.resolve()
