"""`dbsprout train` Typer subcommand."""

from __future__ import annotations

from pathlib import Path

import typer

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
        typer.echo(
            f"Error: train extract requires privacy tier 'local' "
            f"(current: {config.privacy.tier}). "
            f'Set [privacy] tier = "local" in dbsprout.toml.',
            err=True,
        )
        raise typer.Exit(code=2)

    # Lazy import: keeps polars/sqlalchemy off the <500 ms CLI startup path.
    from dbsprout.train.extractor import SampleExtractor  # noqa: PLC0415
    from dbsprout.train.models import ExtractorConfig  # noqa: PLC0415

    extractor = SampleExtractor()
    result = extractor.extract(
        source=db,
        config=ExtractorConfig(
            db_url=db,
            sample_rows=sample_rows,
            output_dir=output,
            seed=seed,
            min_per_table=min_per_table,
            max_per_table=max_per_table,
            quiet=quiet,
        ),
    )
    if not quiet:
        total = sum(r.sampled + r.fk_closure_added for r in result.tables)
        closure = sum(r.fk_closure_added for r in result.tables)
        typer.echo(
            f"Extracted {total} rows across {len(result.tables)} tables "
            f"(FK closure added {closure} rows) in {result.duration_seconds:.2f} s. "
            f"Manifest: {result.manifest_path}"
        )
