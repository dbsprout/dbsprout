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


@train_app.command("serialize")
def serialize(
    input_dir: Path = typer.Option(
        Path(".dbsprout/training"),
        "--input",
        "-i",
        help="Directory produced by 'dbsprout train extract' (contains samples/).",
    ),
    output: Path = typer.Option(
        Path(".dbsprout/training/data.jsonl"),
        "--output",
        "-o",
        help="Destination JSONL training corpus.",
    ),
    seed: int = typer.Option(0, "--seed"),
    null_policy: str = typer.Option(
        "skip",
        "--null-policy",
        help="How NULL cells are rendered: 'skip' (omit clause) or 'literal' (col is NULL).",
    ),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Serialize extracted Parquet samples into GReaT-style JSONL.

    Each row becomes ``[<table>] col is value, ...`` with a seeded per-row
    column shuffle, written as ``{"text": ..., "table": ...}`` JSONL.
    """
    config = load_config()
    if config.privacy.tier != "local":
        console.print(
            f"[red]Error:[/red] train serialize requires privacy tier 'local' "
            f"(current: {config.privacy.tier}). "
            f'Set [privacy] tier = "local" in dbsprout.toml.',
            style="bold",
        )
        raise typer.Exit(code=2)

    # Lazy import: keeps polars off the <500 ms CLI startup path.
    from dbsprout.train.models import NullPolicy  # noqa: PLC0415
    from dbsprout.train.serializer import DataPreparer  # noqa: PLC0415

    try:
        policy = NullPolicy(null_policy)
    except ValueError as exc:
        console.print(
            f"[red]Error:[/red] invalid --null-policy {null_policy!r}; "
            f"expected 'skip' or 'literal'."
        )
        raise typer.Exit(code=2) from exc

    preparer = DataPreparer()
    try:
        result = preparer.prepare(
            input_dir=input_dir,
            output_path=output,
            seed=seed,
            null_policy=policy,
            quiet=quiet,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # AC-10: --quiet suppresses only the Rich progress bar; the one-line
    # summary is always printed so scripted pipelines still get the result.
    nulls = sum(r.nulls_skipped for r in result.tables)
    console.print(
        f"[green bold]Serialized[/green bold] {result.total_rows} rows from "
        f"{len(result.tables)} tables (skipped {nulls} NULL cells) in "
        f"{result.duration_seconds:.2f} s.\n"
        f"Corpus: [cyan]{result.output_path}[/cyan]"
    )
