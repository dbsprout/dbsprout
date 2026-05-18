"""`dbsprout models` Typer subcommand — list/download/info embedded models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from dbsprout.cli.console import console

if TYPE_CHECKING:
    import httpx

models_app = typer.Typer(
    name="models",
    help="List, download, and inspect embedded GGUF models.",
    no_args_is_help=True,
)


_BINARY_STEP = 1024.0
_UNITS = ("B", "KB", "MB", "GB", "TB")


def _fmt_size(num: int) -> str:
    size = float(num)
    for unit in _UNITS:
        if size < _BINARY_STEP or unit == _UNITS[-1]:
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= _BINARY_STEP
    raise AssertionError  # pragma: no cover - loop always returns at TB


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

    registry_filenames = {e.filename for e in load_registry()}
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
        # Surface base models present on disk but absent from the registry
        # (e.g. a manually-placed GGUF) — otherwise they were invisible.
        elif im.kind == "base" and im.name not in registry_filenames:
            table.add_row(
                im.name,
                "-",
                "-",
                _fmt_size(im.size_bytes),
                "base",
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


def _make_client() -> httpx.Client:
    """Create the httpx client (own seam so tests can swap the transport)."""
    import httpx  # noqa: PLC0415 - keep httpx off the <500ms CLI startup path

    from dbsprout.models.manager import _DOWNLOAD_TIMEOUT_S  # noqa: PLC0415

    return httpx.Client(timeout=_DOWNLOAD_TIMEOUT_S, follow_redirects=True)


@models_app.command("download")
def download_model(
    name: str = typer.Argument(..., help="Registry model name to download."),
    force: bool = typer.Option(False, "--force", help="Re-download if installed."),
) -> None:
    """Download a registry model's GGUF into .dbsprout/models/base/."""
    from rich.progress import (  # noqa: PLC0415
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    from dbsprout.models import ModelManager  # noqa: PLC0415
    from dbsprout.models.manager import DownloadError  # noqa: PLC0415

    mgr = ModelManager()
    try:
        entry = mgr.resolve_entry(name)
    except KeyError:
        valid = ", ".join(e.name for e in mgr.registry())
        console.print(f"[red]Error:[/red] unknown model {name!r}. Valid models: {valid}")
        raise typer.Exit(code=1) from None

    if mgr.is_installed(entry) and not force:
        console.print(
            f"[yellow]{entry.name} is already installed[/yellow] at "
            f"[cyan]{mgr.install_path(entry)}[/cyan] (use --force to re-download)."
        )
        return

    client = _make_client()
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(entry.name, total=entry.size_bytes or None)

        def _cb(done: int, total: int) -> None:
            progress.update(task_id, completed=done, total=total or None)

        try:
            dest = mgr.download(entry, client=client, progress_cb=_cb, force=force)
        except DownloadError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    console.print(f"[green bold]Downloaded[/green bold] {entry.name} -> [cyan]{dest}[/cyan]")
