"""`dbsprout train` Typer subcommand."""

from __future__ import annotations

from pathlib import Path

import typer

from dbsprout.cli.console import console
from dbsprout.config import load_config

train_app = typer.Typer(name="train", help="Training pipeline subcommands.", no_args_is_help=True)


@train_app.command("extract")
def extract(  # noqa: PLR0913 - CLI flags are inherently positional/named
    db: str = typer.Option(..., "--db", envvar="DBSPROUT_TARGET_DB"),
    sample_rows: int = typer.Option(1000, "--sample-rows", min=1),
    output: Path = typer.Option(Path(".dbsprout/training"), "--output", "-o"),
    seed: int = typer.Option(0, "--seed"),
    min_per_table: int = typer.Option(10, "--min-per-table", min=0),
    max_per_table: int | None = typer.Option(None, "--max-per-table"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Extract a stratified sample from a live database into Parquet files."""
    config = load_config()
    if config.privacy.tier != "local":
        console.print(
            f"[red]Error:[/red] train extract requires privacy tier 'local' "
            f"(current: {config.privacy.tier}). "
            f'Set [privacy] tier = "local" in dbsprout.toml.',
            style="bold",
        )
        raise typer.Exit(code=2)

    # Lazy import: keeps polars/sqlalchemy off the <500 ms CLI startup path.
    from dbsprout.cli._utils import scrub_secrets  # noqa: PLC0415
    from dbsprout.train.extractor import SampleExtractor  # noqa: PLC0415
    from dbsprout.train.models import ExtractorConfig  # noqa: PLC0415

    extractor = SampleExtractor()
    cfg = ExtractorConfig(
        sample_rows=sample_rows,
        output_dir=output,
        seed=seed,
        min_per_table=min_per_table,
        max_per_table=max_per_table,
        quiet=quiet,
    )
    try:
        result = extractor.extract(source=db, config=cfg)
    except Exception as exc:
        safe_msg = scrub_secrets(str(exc), db)
        console.print(f"[red]Error:[/red] {safe_msg}")
        raise typer.Exit(code=1) from exc
    if not quiet:
        total = sum(r.sampled + r.fk_closure_added for r in result.tables)
        closure = sum(r.fk_closure_added for r in result.tables)
        console.print(
            f"[green bold]Extracted[/green bold] {total} rows across "
            f"{len(result.tables)} tables (FK closure added {closure} rows) in "
            f"{result.duration_seconds:.2f} s.\n"
            f"Manifest: [cyan]{result.manifest_path}[/cyan]"
        )
