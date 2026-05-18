"""``dbsprout report`` command — render an HTML report from the state DB.

Standalone counterpart to ``dbsprout generate --report``: reads recorded
generation-run telemetry (S-079/S-080) and renders the self-contained HTML
report (S-081) **without regenerating any data**. The report module and its
templates are consumed read-only — this command only orchestrates run
selection and calls :class:`~dbsprout.report.generator.ReportGenerator`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

if TYPE_CHECKING:
    from dbsprout.state import RunRecord

console = Console()

#: Default state-DB location (matches ``state.writer._DEFAULT_DB_PATH`` and
#: ``StateDB``'s own default), resolved relative to the current directory.
_DEFAULT_DB_PATH = Path(".dbsprout") / "state.db"

#: Exact no-runs error string (S-085 acceptance criteria — do not reword).
_NO_RUNS_ERROR = "No generation runs found. Run `dbsprout generate --report` first."


class _SingleRunStateDB:
    """Read-only :class:`StateDB` stand-in exposing exactly one run.

    :class:`~dbsprout.report.generator.ReportGenerator` renders
    ``state_db.get_runs()[0]``. To render a *specific* historic run via
    ``--run-id`` without modifying the report module/templates, we hand it
    this duck-typed adapter whose ``get_runs()`` returns just the selected
    run. This is the minimal scope-respecting seam (S-085).
    """

    def __init__(self, run: RunRecord) -> None:
        self._run = run

    def get_runs(self) -> list[RunRecord]:
        return [self._run]


def _select_run(runs: list[RunRecord], run_id: int | None) -> RunRecord:
    """Pick the newest run, or the one matching ``run_id``.

    ``runs`` is newest-first (``StateDB.get_runs`` contract). Raises
    ``typer.Exit(1)`` with a clear message when ``run_id`` matches nothing.
    """
    if run_id is None:
        return runs[0]
    for run in runs:
        if run.id == run_id:
            return run
    console.print(
        f"[red]Error:[/red] Generation run {run_id} not found. "
        f"Available run ids: {', '.join(str(r.id) for r in runs)}."
    )
    raise typer.Exit(code=1)


def report_command(
    output: Path | None = None,
    run_id: int | None = None,
    db_path: Path | None = None,
) -> None:
    """Render an HTML report from the most recent (or a specific) run."""
    from dbsprout.config.models import DBSproutConfig  # noqa: PLC0415
    from dbsprout.report.generator import ReportGenerator  # noqa: PLC0415
    from dbsprout.state import StateDB  # noqa: PLC0415

    state_db_path = db_path if db_path is not None else _DEFAULT_DB_PATH
    if not state_db_path.exists():
        console.print(f"[red]Error:[/red] {_NO_RUNS_ERROR}")
        raise typer.Exit(code=1)

    runs = StateDB(state_db_path).get_runs()
    if not runs:
        console.print(f"[red]Error:[/red] {_NO_RUNS_ERROR}")
        raise typer.Exit(code=1)

    selected = _select_run(runs, run_id)

    if output is not None:
        out_path = output
    else:
        cfg_path = Path("dbsprout.toml")
        config = DBSproutConfig.from_toml(cfg_path if cfg_path.exists() else None)
        out_path = Path(config.report.output)

    generator = ReportGenerator(out_path)
    # Newest run → pass the real StateDB; a specific older run → adapter.
    source = StateDB(state_db_path) if selected.id == runs[0].id else _SingleRunStateDB(selected)
    written = generator.generate(source)  # type: ignore[arg-type]
    console.print(f"[green]Report saved to[/green] {written}")
