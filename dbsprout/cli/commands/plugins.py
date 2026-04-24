"""``dbsprout plugins`` — inspect discovered plugins."""

from __future__ import annotations

import typer
from rich.table import Table

from dbsprout.cli.console import console

plugins_app = typer.Typer(
    name="plugins",
    help="Inspect discovered dbsprout plugins.",
    no_args_is_help=True,
)


@plugins_app.command("list")
def list_cmd() -> None:
    """List discovered plugins grouped by entry-point group."""
    from dbsprout.plugins.registry import get_registry  # noqa: PLC0415

    registry = get_registry()
    table = Table(title="DBSprout plugins")
    table.add_column("group")
    table.add_column("name")
    table.add_column("module")
    table.add_column("status")
    for info in sorted(registry.list(), key=lambda i: (i.group, i.name)):
        table.add_row(info.group, info.name, info.module, info.status)
    console.print(table)


@plugins_app.command("check")
def check_cmd(target: str) -> None:
    """Validate ``<group>:<name>`` loads and conforms to its Protocol.

    Exits 0 on success, 2 on failure with a diagnostic message.
    """
    from dbsprout.plugins.errors import PluginValidationError  # noqa: PLC0415
    from dbsprout.plugins.registry import get_registry  # noqa: PLC0415

    if ":" not in target:
        console.print(
            "[red]Error:[/red] target must be '<group>:<name>' (e.g. 'dbsprout.parsers:dbml')."
        )
        raise typer.Exit(code=2)
    group, name = target.split(":", 1)

    try:
        info = get_registry().check(group, name)
    except PluginValidationError as exc:
        console.print(f"[red]Invalid:[/red] {exc}")
        raise typer.Exit(code=2) from None
    console.print(f"[green]OK:[/green] {info.group}:{info.name} @ {info.module}")
