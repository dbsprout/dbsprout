"""DBSprout CLI entry point."""

from __future__ import annotations

import typer

from dbsprout.cli.commands.generate import generate_command
from dbsprout.cli.commands.init import init_command

app = typer.Typer(
    name="dbsprout",
    help="Generate realistic seed data from your database schema.",
    no_args_is_help=True,
)

app.command(name="init")(init_command)
app.command(name="generate")(generate_command)


@app.callback()
def main() -> None:
    """DBSprout — realistic database seed data from your schema."""
