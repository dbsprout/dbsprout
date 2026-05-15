"""`dbsprout models` Typer subcommand — list/download/info embedded models."""

from __future__ import annotations

import typer

from dbsprout.cli.console import console

models_app = typer.Typer(
    name="models",
    help="List, download, and inspect embedded GGUF models.",
    no_args_is_help=True,
)


_BINARY_STEP = 1024.0


def _fmt_size(num: int) -> str:
    size = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < _BINARY_STEP or unit == "TB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= _BINARY_STEP
    return f"{size:.1f} TB"


@models_app.command("list")
def list_models() -> None:
    """List registry models and locally installed base/custom models."""
    from rich.table import Table  # noqa: PLC0415

    from dbsprout.models import ModelManager, load_registry  # noqa: PLC0415

    mgr = ModelManager()
    installed = {im.name: im for im in mgr.list_installed()}

    table = Table(title="DBSprout Models")
    table.add_column("Name")
    table.add_column("Params")
    table.add_column("Quant")
    table.add_column("Size", justify="right")
    table.add_column("Kind")
    table.add_column("Installed")

    for entry in load_registry():
        table.add_row(
            entry.name,
            entry.parameters,
            entry.quantization,
            _fmt_size(entry.size_bytes),
            "base",
            "[green]yes[/green]" if mgr.is_installed(entry) else "no",
        )
    for im in installed.values():
        if im.kind == "custom":
            table.add_row(
                im.name,
                "-",
                "-",
                _fmt_size(im.size_bytes),
                "custom",
                "[green]yes[/green]",
            )

    console.print(table)


@models_app.command("info")
def model_info(name: str = typer.Argument(..., help="Registry model name.")) -> None:
    """Show details for one registry model."""
    from rich.panel import Panel  # noqa: PLC0415

    from dbsprout.models import ModelManager  # noqa: PLC0415

    mgr = ModelManager()
    try:
        entry = mgr.resolve_entry(name)
    except KeyError:
        valid = ", ".join(e.name for e in mgr.registry())
        console.print(f"[red]Error:[/red] unknown model {name!r}. Valid models: {valid}")
        raise typer.Exit(code=1) from None

    installed = mgr.is_installed(entry)
    install_line = f"yes ({mgr.install_path(entry)})" if installed else "no"
    body = (
        f"[bold]{entry.name}[/bold]\n{entry.description}\n\n"
        f"Parameters  : {entry.parameters}\n"
        f"Quantization: {entry.quantization}\n"
        f"Size        : {_fmt_size(entry.size_bytes)}\n"
        f"Source      : {entry.repo} / {entry.filename}\n"
        f"Training    : {entry.training}\n"
        f"License     : {entry.license or '-'}\n"
        f"Installed   : {install_line}"
    )
    console.print(Panel(body, title=f"Model: {entry.name}"))
