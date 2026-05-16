"""Render :class:`DBSproutError` as a Rich panel and bridge to Typer (S-076).

This is *additive*: existing ``console.print("[red]Error:[/red] …")`` +
``raise typer.Exit(...)`` call sites are unaffected. The handler is the single
chokepoint for any :class:`DBSproutError` that propagates out of a command.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import typer
from rich.panel import Panel
from rich.text import Text

from dbsprout.cli._utils import scrub_secrets
from dbsprout.cli.console import console
from dbsprout.errors import DBSproutError

if TYPE_CHECKING:
    from collections.abc import Iterator


def format_error_panel(exc: DBSproutError, *, source_url: str | None = None) -> Panel:
    """Build a colour-coded Rich panel with What / Why / Fix rows."""

    def _clean(value: str) -> str:
        return scrub_secrets(value, source_url) if source_url else value

    body = Text()
    body.append("What: ", style="bold red")
    body.append(_clean(exc.what) + "\n")
    body.append("Why:  ", style="bold yellow")
    body.append(_clean(exc.why) + "\n")
    body.append("Fix:  ", style="bold green")
    body.append(_clean(exc.fix))
    return Panel(
        body,
        title="[bold red]DBSprout Error[/bold red]",
        border_style="red",
        expand=False,
    )


@contextmanager
def handle_cli_errors(*, verbose: bool) -> Iterator[None]:
    """Catch :class:`DBSproutError` inside the block and render it.

    On a :class:`DBSproutError`: print the panel and re-raise
    ``typer.Exit(exc.exit_code)``. When *verbose*, also print the traceback.
    Any other exception is re-raised when *verbose*; otherwise it is wrapped in
    a generic panel and surfaced as ``typer.Exit(1)``.
    """
    try:
        yield
    except DBSproutError as exc:
        console.print(format_error_panel(exc))
        if verbose:
            console.print_exception()
        raise typer.Exit(code=exc.exit_code) from None
    except typer.Exit:
        raise
    except Exception as exc:  # top-level CLI guard — render any error nicely
        if verbose:
            raise
        # Do not echo the raw exception text: messages from libraries such as
        # SQLAlchemy can embed connection URLs (with passwords). Only the
        # exception class is safe to show; --verbose surfaces the full detail.
        generic = DBSproutError(
            what="An unexpected error occurred.",
            why=f"Unhandled {type(exc).__name__}.",
            fix="Re-run with --verbose for the full traceback.",
        )
        console.print(format_error_panel(generic))
        raise typer.Exit(code=1) from None
