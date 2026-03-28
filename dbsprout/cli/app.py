"""DBSprout CLI entry point."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="dbsprout",
    help="Generate realistic seed data from your database schema.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """DBSprout — realistic database seed data from your schema."""
