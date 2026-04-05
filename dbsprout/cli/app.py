"""DBSprout CLI entry point.

Commands are lazy-imported to avoid pulling in heavy dependencies
(sqlalchemy, mimesis, numpy) at startup.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="dbsprout",
    help="Generate realistic seed data from your database schema.",
    no_args_is_help=True,
)


@app.command(name="init")
def init_proxy(
    db: str | None = typer.Option(None, "--db", help="Database URL."),
    file: str | None = typer.Option(None, "--file", help="DDL file."),
    output_dir: str = typer.Option(".", "--output-dir", "-o"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Introspect a database schema and generate configuration."""
    from pathlib import Path  # noqa: PLC0415

    from dbsprout.cli.commands.init import init_command  # noqa: PLC0415

    init_command(
        db=db,
        file=file,
        output_dir=Path(output_dir),
        dry_run=dry_run,
    )


@app.command(name="generate")
def generate_proxy(  # noqa: PLR0913
    schema_snapshot: str | None = typer.Option(
        None,
        "--schema-snapshot",
    ),
    config_path: str | None = typer.Option(None, "--config"),
    rows: int = typer.Option(100, "--rows", "-n", min=1),
    seed: int = typer.Option(42, "--seed", "-s", min=0),
    output_format: str = typer.Option(
        "sql", "--output-format", "-f", help="Output format: sql, csv, json, jsonl, direct."
    ),
    output_dir: str = typer.Option("./seeds", "--output-dir", "-o"),
    dialect: str = typer.Option("postgresql", "--dialect", "-d"),
    engine: str = typer.Option("heuristic", "--engine", "-e"),
    privacy: str = typer.Option("local", "--privacy"),
    db: str | None = typer.Option(None, "--db", help="Target database URL for direct insertion."),
) -> None:
    """Generate seed data from a schema snapshot."""
    from pathlib import Path  # noqa: PLC0415

    from dbsprout.cli.commands.generate import generate_command  # noqa: PLC0415

    generate_command(
        schema_snapshot=Path(schema_snapshot) if schema_snapshot else None,
        config_path=Path(config_path) if config_path else None,
        rows=rows,
        seed=seed,
        output_format=output_format,
        output_dir=Path(output_dir),
        dialect=dialect,
        engine=engine,
        privacy=privacy,
        target_db=db,
    )


@app.command(name="validate")
def validate_proxy(  # noqa: PLR0913
    schema_snapshot: str | None = typer.Option(
        None,
        "--schema-snapshot",
    ),
    config_path: str | None = typer.Option(None, "--config"),
    rows: int = typer.Option(100, "--rows", "-n", min=1),
    seed: int = typer.Option(42, "--seed", "-s", min=0),
    output_format: str = typer.Option("rich", "--format", "-f"),
    engine: str = typer.Option("heuristic", "--engine", "-e"),
) -> None:
    """Validate integrity of generated seed data."""
    from pathlib import Path  # noqa: PLC0415

    from dbsprout.cli.commands.validate import validate_command  # noqa: PLC0415

    validate_command(
        schema_snapshot=Path(schema_snapshot) if schema_snapshot else None,
        config_path=Path(config_path) if config_path else None,
        rows=rows,
        seed=seed,
        output_format=output_format,
        engine=engine,
    )


@app.command(name="audit")
def audit_proxy(
    last: int | None = typer.Option(None, "--last", "-n", min=1),
) -> None:
    """Show the LLM interaction audit log."""
    from dbsprout.cli.commands.audit import audit_command  # noqa: PLC0415

    audit_command(last=last)


@app.callback()
def main() -> None:
    """DBSprout — realistic database seed data from your schema."""
