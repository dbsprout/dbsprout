"""`dbsprout train` Typer subcommand."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from dbsprout.cli.console import console
from dbsprout.config import load_config

if TYPE_CHECKING:
    from collections.abc import Callable

# NOTE: no_args_is_help is intentionally NOT set — the `pipeline` callback below
# uses invoke_without_command=True and renders help itself for the bare
# `dbsprout train` case (the two options are mutually exclusive in Typer).
train_app = typer.Typer(name="train", help="Training pipeline subcommands.")


def _privacy_gate(stage: str) -> None:
    """Refuse to run *stage* unless the privacy tier is ``local``.

    Training on real database rows must stay on the local machine. Mirrors the
    gate in ``extract``/``serialize``/``run`` so the end-to-end pipeline is
    held to the same bar. Exits with code 2 (config error) on a violation.
    """
    config = load_config()
    if config.privacy.tier != "local":
        console.print(
            f"[red]Error:[/red] train {stage} requires privacy tier 'local' "
            f"(current: {config.privacy.tier}). "
            f'Set [privacy] tier = "local" in dbsprout.toml.',
            style="bold",
        )
        raise typer.Exit(code=2)


@train_app.callback(invoke_without_command=True)
def pipeline(  # noqa: PLR0913 - CLI flags are inherently positional/named
    ctx: typer.Context,
    db: str | None = typer.Option(
        None, "--db", envvar="DBSPROUT_TARGET_DB", help="Live database URL to sample from."
    ),
    sample_rows: int = typer.Option(1000, "--sample-rows", min=1),
    epochs: int | None = typer.Option(None, "--epochs", min=1),
    output: Path = typer.Option(
        Path(".dbsprout"),
        "--output",
        "-o",
        help="Base directory for training artifacts (samples, corpus, adapter, GGUF).",
    ),
    seed: int = typer.Option(0, "--seed"),
    no_pii_redaction: bool = typer.Option(
        False,
        "--no-pii-redaction",
        help="Disable PII value redaction before serialization (non-sensitive data only).",
    ),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Run the full fine-tuning pipeline: extract -> serialize -> train -> export.

    Auto-detects the backend (CUDA -> Unsloth, Apple Silicon -> MLX) and writes
    a quantized GGUF model ready for the embedded provider. Sub-commands
    (``extract``/``serialize``/``run``) still work for running a single stage.
    """
    # A subcommand was given (e.g. `dbsprout train extract`): defer to it.
    if ctx.invoked_subcommand is not None:
        return
    # Bare `dbsprout train` with no --db: fall through to Typer's help.
    if db is None:
        console.print(ctx.get_help())
        return

    _privacy_gate("pipeline")
    _run_pipeline(
        db=db,
        sample_rows=sample_rows,
        epochs=epochs,
        output=output,
        seed=seed,
        quiet=quiet,
        no_pii_redaction=no_pii_redaction,
    )


@dataclass(frozen=True)
class _PipelineComponents:
    """Injectable seam for the four pipeline stages (S-068 review #16).

    Holds factories/callables instead of importing the heavy stage modules at
    the call site. Tests replace this whole object so the suite runs without
    the ``[data]``/``[stats]`` extras (no real ``polars``/``torch``/``mlx``
    import); production uses :func:`_default_components`, which lazily imports
    the real implementations only when the pipeline actually runs.
    """

    make_extractor: Callable[[], Any]
    make_redactor: Callable[[], Any]
    make_preparer: Callable[[], Any]
    select_trainer: Callable[[], Any]
    make_exporter: Callable[[], Any]


def _default_components() -> _PipelineComponents:
    """Build the real pipeline components (heavy deps lazily imported here)."""
    # Lazy import: keeps polars/torch/unsloth/mlx off the <500 ms startup path
    # AND off plain ``import dbsprout.cli.commands.train`` (so core-only test
    # collection never pulls the optional extras).
    from dbsprout.train.exporter import Exporter  # noqa: PLC0415
    from dbsprout.train.extractor import SampleExtractor  # noqa: PLC0415
    from dbsprout.train.mlx_trainer import select_trainer  # noqa: PLC0415
    from dbsprout.train.privacy import TrainingRedactor  # noqa: PLC0415
    from dbsprout.train.serializer import DataPreparer  # noqa: PLC0415

    return _PipelineComponents(
        make_extractor=SampleExtractor,
        make_redactor=TrainingRedactor,
        make_preparer=DataPreparer,
        select_trainer=select_trainer,
        make_exporter=Exporter,
    )


def _apply_privacy(  # noqa: PLR0913 - explicit deps keep the seam testable
    privacy_cfg: Any,
    *,
    comps: _PipelineComponents,
    sample_dir: Path,
    db: str,
    no_pii_redaction: bool,
    scrub_secrets: Callable[[str, str], str],
) -> None:
    """Run privacy layers 3 + 4 between extract and serialize (S-070).

    Enforces the DP-SGD guard, then (unless ``--no-pii-redaction`` or the
    ``[train.privacy]`` config disables it) redacts the extracted Parquet
    in place via the injected ``make_redactor`` seam so the serializer reads
    masked values. All errors are scrubbed of DB credentials and mapped to
    exit code 1, mirroring the four stage handlers.
    """
    from dbsprout.train.privacy import dp_sgd_guard  # noqa: PLC0415

    try:
        dp_sgd_guard(privacy_cfg)
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {scrub_secrets(str(exc), db)}")
        raise typer.Exit(code=1) from exc

    if no_pii_redaction:
        privacy_cfg = privacy_cfg.model_copy(update={"pii_redaction": False})
    if not privacy_cfg.pii_redaction:
        return

    console.print("[bold blue]\\[1.5/4][/bold blue] Redacting PII before serialization...")
    try:
        redaction_stats = comps.make_redactor().redact_dir(sample_dir, config=privacy_cfg)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {scrub_secrets(str(exc), db)}")
        raise typer.Exit(code=1) from exc
    _print_redaction_summary(redaction_stats)


def _run_pipeline(  # noqa: PLR0913 - one knob per pipeline stage
    *,
    db: str,
    sample_rows: int,
    epochs: int | None,
    output: Path,
    seed: int,
    quiet: bool,
    no_pii_redaction: bool = False,
    components: _PipelineComponents | None = None,
) -> None:
    """Glue the four pipeline stages together via the injectable seam."""
    from dbsprout.cli._utils import scrub_secrets  # noqa: PLC0415
    from dbsprout.train.models import ExtractorConfig, NullPolicy  # noqa: PLC0415

    comps = components if components is not None else _default_components()

    config = load_config()
    train_overrides = {k: v for k, v in {"epochs": epochs}.items() if v is not None}
    train_config = config.train.model_copy(update=train_overrides)

    sample_dir = output / "training"
    corpus_path = sample_dir / "data.jsonl"
    adapter_dir = output / "models" / "adapters"
    gguf_dir = output / "models" / "custom"

    # Stage 1: extract a stratified sample from the live database.
    # ``scrub_secrets(str(exc), db)`` is applied uniformly to ALL four stage
    # handlers (defense-in-depth, S-068 review #17): any stage's exception
    # message could embed the DB URL/credentials transitively.
    console.print("[bold blue]\\[1/4][/bold blue] Extracting sample from database...")
    try:
        extract_result = comps.make_extractor().extract(
            source=db,
            config=ExtractorConfig(
                sample_rows=sample_rows, output_dir=sample_dir, seed=seed, quiet=quiet
            ),
        )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {scrub_secrets(str(exc), db)}")
        raise typer.Exit(code=1) from exc
    samples = sum(t.sampled + t.fk_closure_added for t in extract_result.tables)

    # Privacy layers 3 + 4 (S-070): DP-SGD guard, then PII value redaction of
    # the extracted Parquet *before* serialization (the S-063 dependency).
    _apply_privacy(
        train_config.privacy,
        comps=comps,
        sample_dir=sample_dir,
        db=db,
        no_pii_redaction=no_pii_redaction,
        scrub_secrets=scrub_secrets,
    )

    # Stage 2: serialize Parquet samples into GReaT-style JSONL.
    console.print("[bold blue]\\[2/4][/bold blue] Serializing rows to JSONL corpus...")
    try:
        serialize_result = comps.make_preparer().prepare(
            input_dir=sample_dir,
            output_path=corpus_path,
            seed=seed,
            null_policy=NullPolicy.SKIP,
            quiet=quiet,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {scrub_secrets(str(exc), db)}")
        raise typer.Exit(code=1) from exc

    # Stage 3: fine-tune a LoRA adapter (CUDA or MLX, auto-detected).
    console.print("[bold blue]\\[3/4][/bold blue] Fine-tuning LoRA adapter...")
    try:
        trainer = comps.select_trainer()
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {scrub_secrets(str(exc), db)}")
        raise typer.Exit(code=1) from exc
    try:
        adapter = trainer.train(
            corpus_path=serialize_result.output_path,
            config=train_config,
            output_dir=adapter_dir,
            quiet=quiet,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {scrub_secrets(str(exc), db)}")
        raise typer.Exit(code=1) from exc

    # Stage 4: merge + quantize to a GGUF model.
    console.print("[bold blue]\\[4/4][/bold blue] Exporting GGUF model...")
    try:
        export_result = comps.make_exporter().to_gguf_result(
            adapter,
            Path(train_config.base_model),
            output_dir=gguf_dir,
            quiet=quiet,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {scrub_secrets(str(exc), db)}")
        raise typer.Exit(code=1) from exc

    _print_summary(
        samples=samples,
        serialize_result=serialize_result,
        adapter=adapter,
        export_result=export_result,
    )


def _print_redaction_summary(stats: Any) -> None:
    """One-line PII-redaction summary (always shown, even with --quiet).

    When Presidio is not installed the redactor reports
    ``presidio_available=False``; surface that clearly so the operator knows
    layer-3 redaction did NOT run rather than assuming the data was masked.
    """
    if not stats.presidio_available:
        console.print(
            "[yellow]PII redaction skipped:[/yellow] Presidio not installed "
            "(install 'dbsprout[privacy]' or set [train.privacy] "
            "pii_redaction = false to silence)."
        )
        return
    types = ", ".join(sorted(stats.entity_totals)) or "none"
    console.print(
        f"[green]Redacted[/green] {stats.total_values_masked} PII value(s) (types: {types})."
    )


def _human_size(num: int) -> str:
    """Format a byte count as a human-readable size (binary units)."""
    size = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError  # pragma: no cover - loop always returns at TB


def _print_summary(
    *,
    samples: int,
    serialize_result: object,
    adapter: object,
    export_result: object,
) -> None:
    """Print the final one-shot pipeline summary (always shown, even quiet)."""
    rows = getattr(serialize_result, "total_rows", samples)
    duration = getattr(adapter, "duration_seconds", 0.0)
    gguf_path = getattr(export_result, "gguf_path", "")
    size_bytes = getattr(export_result, "size_bytes", 0)
    console.print(
        f"[green bold]Pipeline complete.[/green bold]\n"
        f"  Samples extracted : {samples} ({rows} serialized rows)\n"
        f"  Training time     : {duration:.1f} s\n"
        f"  Adapter           : [cyan]{getattr(adapter, 'adapter_path', '')}[/cyan]\n"
        f"  GGUF model        : [cyan]{gguf_path}[/cyan] "
        f"({_human_size(size_bytes)})"
    )


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


@train_app.command("run")
def run(  # noqa: PLR0913 - CLI flags are inherently positional/named
    corpus: Path = typer.Option(
        Path(".dbsprout/training/data.jsonl"),
        "--corpus",
        "-c",
        help="JSONL corpus produced by 'dbsprout train serialize'.",
    ),
    output: Path = typer.Option(
        Path(".dbsprout/models/adapters"),
        "--output",
        "-o",
        help="Directory to write the LoRA adapter into.",
    ),
    schema_hash: str | None = typer.Option(
        None, "--schema-hash", help="Subdirectory name for this schema's adapter."
    ),
    epochs: int | None = typer.Option(None, "--epochs", min=1),
    learning_rate: float | None = typer.Option(None, "--learning-rate", min=0.0),
    lora_rank: int | None = typer.Option(None, "--lora-rank", min=1),
    lora_alpha: int | None = typer.Option(None, "--lora-alpha", min=1),
    batch_size: int | None = typer.Option(None, "--batch-size", min=1),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Fine-tune a QLoRA adapter on a serialized training corpus.

    Auto-detects CUDA and uses the Unsloth backend. Requires an NVIDIA GPU
    and ``pip install dbsprout[train-cuda]``.
    """
    config = load_config()
    if config.privacy.tier != "local":
        console.print(
            f"[red]Error:[/red] train run requires privacy tier 'local' "
            f"(current: {config.privacy.tier}). "
            f'Set [privacy] tier = "local" in dbsprout.toml.',
            style="bold",
        )
        raise typer.Exit(code=2)

    # Lazy import: keeps torch/unsloth off the <500 ms CLI startup path.
    from dbsprout.train.trainer import QLoRATrainer  # noqa: PLC0415

    overrides = {
        k: v
        for k, v in {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "batch_size": batch_size,
        }.items()
        if v is not None
    }
    train_config = config.train.model_copy(update=overrides)

    trainer = QLoRATrainer()
    try:
        adapter = trainer.train(
            corpus_path=corpus,
            config=train_config,
            output_dir=output,
            schema_hash=schema_hash,
            quiet=quiet,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    loss = f"{adapter.final_loss:.4f}" if adapter.final_loss is not None else "n/a"
    console.print(
        f"[green bold]Trained[/green bold] QLoRA adapter on "
        f"{adapter.train_samples} samples ({adapter.epochs} epochs, "
        f"final loss {loss}) in {adapter.duration_seconds:.1f} s.\n"
        f"Adapter: [cyan]{adapter.adapter_path}[/cyan]"
    )
