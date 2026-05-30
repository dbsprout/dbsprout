"""Core service facade — the single orchestration seam (S-106).

Both the CLI (today) and the future web UI (Epic A) call THESE functions so
the generation / validation / output pipeline is never duplicated or forked.
The facade COMPOSES the existing stage modules; it does not reimplement them
and it does not change ``orchestrate()`` semantics (S-106 is a refactor for
byte-parity, not a behaviour change).

UI-agnostic contract
--------------------
Functions take plain / Pydantic arguments and return frozen result objects or
raise plain exceptions (``ValueError`` and the domain ``dbsprout.errors.*``).
This module imports NO ``typer``, ``rich``, or ``fastapi``. Where the CLI used
to interleave a ``console.print`` warning with a pipeline step, the relevant
facade function returns the warning *string* in its result object and the
caller decides how to surface it. (Live progress callbacks / cooperative
cancellation are now wired in via S-107: ``generate`` forwards an optional
``progress_callback`` and ``cancel_token`` to ``orchestrate``.)

Stage layering (``schema/ <- spec/ <- generate/ -> output/``) is unchanged: the
facade sits ABOVE the stages, exactly where the CLI sits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dbsprout.generate.orchestrator import GenerateResult, orchestrate
from dbsprout.plugins.dispatch import resolve_writer
from dbsprout.quality.integrity import IntegrityReport, validate_integrity

if TYPE_CHECKING:
    from collections.abc import Callable

    from dbsprout.cli.sources import SchemaSource
    from dbsprout.config.models import DBSproutConfig
    from dbsprout.generate.progress import CancelToken, ProgressEvent
    from dbsprout.quality.detection import DetectionReport
    from dbsprout.quality.fidelity import FidelityReport
    from dbsprout.schema.models import DatabaseSchema

# File-format writers the facade dispatches. The live-DB ``direct`` target is
# intentionally NOT here: it is an output-*target* adapter coupled to CLI
# fallback UX (psycopg / pymysql ImportError handling, per-dialect warnings)
# and stays in the CLI. See the S-106 design doc.
_FILE_FORMATS = frozenset({"sql", "csv", "json", "jsonl", "parquet"})

__all__ = [
    "ConnectionProbe",
    "GenerateResult",
    "IntegrityReport",
    "ValidationOutcome",
    "WriteOutcome",
    "generate",
    "load_schema",
    "probe_connection",
    "run_validation",
    "validate_integrity",
    "write_output",
]


@dataclass(frozen=True)
class ConnectionProbe:
    """Result of a lightweight connection test (no column introspection)."""

    dialect: str
    server_version: str
    table_count: int
    latency_ms: int


def probe_connection(url: str) -> ConnectionProbe:
    """Open *url*, report dialect/version/table-count/latency, then dispose.

    Reuses the validated engine factory from introspection (dialect allow-list +
    timeout). Raises the underlying SQLAlchemy/ValueError on failure; the web
    layer classifies it into a typed envelope.
    """
    import time  # noqa: PLC0415

    import sqlalchemy as sa  # noqa: PLC0415

    from dbsprout.schema.introspect import _create_engine, _validate_url  # noqa: PLC0415

    _validate_url(url)
    start = time.perf_counter()
    engine = _create_engine(url)
    try:
        with engine.connect() as conn:
            inspector = sa.inspect(conn)
            table_count = len(inspector.get_table_names())
            version_info: tuple[object, ...] | None = engine.dialect.server_version_info
        dialect = engine.dialect.name
    finally:
        engine.dispose()
    latency_ms = int((time.perf_counter() - start) * 1000)
    server_version = ".".join(str(p) for p in version_info) if version_info else "unknown"
    return ConnectionProbe(
        dialect=dialect,
        server_version=server_version,
        table_count=table_count,
        latency_ms=latency_ms,
    )


@dataclass(frozen=True)
class WriteOutcome:
    """Result of :func:`write_output`.

    ``warnings`` carries non-fatal messages the caller may surface (e.g. an
    ``--upsert`` flag ignored for a format that does not support it). Returning
    them as data keeps the facade UI-agnostic — the CLI prints them, the web
    UI can render them however it likes.
    """

    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationOutcome:
    """Result of :func:`run_validation`.

    Bundles the generated rows plus every quality report the caller requested.
    ``fidelity`` / ``detection`` are ``None`` when not computed. The caller
    decides how to render the reports and which ones gate the exit code.
    """

    integrity: IntegrityReport
    tables_data: dict[str, list[dict[str, Any]]]
    fidelity: FidelityReport | None = None
    detection: DetectionReport | None = None


def load_schema(source: SchemaSource) -> DatabaseSchema:
    """Load a unified :class:`DatabaseSchema` from a DB URL or a schema file.

    Introspects a live database when ``source.kind == "db"``, otherwise parses
    the file at ``source.raw_value``. Raises the underlying loader exceptions
    (``FileNotFoundError``, ``ValueError``, ``OSError``,
    ``sqlalchemy.exc.SQLAlchemyError``); callers translate them into their own
    error surface (the CLI scrubs secrets and maps to ``typer.Exit(2)``).
    """
    from dbsprout.schema.introspect import introspect  # noqa: PLC0415
    from dbsprout.schema.parsers import parse_schema_file  # noqa: PLC0415

    if source.kind == "db":
        return introspect(source.raw_value)
    return parse_schema_file(Path(source.raw_value))


def generate(  # noqa: PLR0913
    schema: DatabaseSchema,
    config: DBSproutConfig,
    *,
    seed: int,
    default_rows: int,
    engine: str = "heuristic",
    reference_data: dict[str, list[dict[str, Any]]] | None = None,
    lora_path: Path | None = None,
    progress_callback: Callable[[ProgressEvent], None] | None = None,
    cancel_token: CancelToken | Callable[[], bool] | None = None,
) -> GenerateResult:
    """Run the full generation pipeline (the single seam over ``orchestrate``).

    A thin pass-through so every surface shares one generation entry point.
    Semantics are byte-identical to calling ``orchestrate`` directly.

    ``progress_callback`` and ``cancel_token`` (S-107) are optional live-progress
    / cooperative-cancel hooks forwarded verbatim to ``orchestrate`` so a caller
    (the web UI) can stream progress and cancel a running job. Both default to
    ``None``; when unset, output is byte-identical (parity).
    """
    return orchestrate(
        schema,
        config,
        seed=seed,
        default_rows=default_rows,
        engine=engine,
        reference_data=reference_data,
        lora_path=lora_path,
        progress_callback=progress_callback,
        cancel_token=cancel_token,
    )


def write_output(  # noqa: PLR0913
    result: GenerateResult,
    schema: DatabaseSchema,
    insertion_order: list[str],
    output_dir: Path,
    *,
    output_format: str,
    dialect: str = "postgresql",
    upsert: bool = False,
) -> WriteOutcome:
    """Write generated data with the selected file-format writer.

    Handles the non-interactive formats (``sql``, ``csv``, ``json``, ``jsonl``,
    ``parquet``). The live-DB ``direct`` target is NOT handled here — it stays
    in the CLI (see module docstring). Raises ``ValueError`` for an unknown or
    unsupported (``direct``) format.
    """
    if output_format not in _FILE_FORMATS:
        msg = f"Unknown or unsupported output format for write_output: {output_format!r}"
        raise ValueError(msg)

    warnings: list[str] = []
    if upsert and output_format != "sql":
        # Byte-identical to the original CLI warning (minus the Rich "Warning:"
        # prefix, which the caller adds) so console output is unchanged.
        warnings.append(
            "--upsert only applies to --output-format sql or direct; "
            f"it is ignored for {output_format!r}."
        )

    if output_format == "sql":
        resolve_writer("sql").write(
            result.tables_data,
            schema,
            insertion_order,
            output_dir,
            dialect=dialect,
            upsert=upsert,
        )
    elif output_format in ("json", "jsonl"):
        resolve_writer(output_format).write(
            result.tables_data,
            schema,
            insertion_order,
            output_dir,
            fmt=output_format,
        )
    else:  # csv, parquet
        resolve_writer(output_format).write(result.tables_data, schema, insertion_order, output_dir)

    return WriteOutcome(warnings=tuple(warnings))


def run_validation(  # noqa: PLR0913
    schema: DatabaseSchema,
    config: DBSproutConfig,
    *,
    seed: int,
    default_rows: int,
    engine: str = "heuristic",
    reference_data: dict[str, list[dict[str, Any]]] | None = None,
    detection: bool = False,
) -> ValidationOutcome:
    """Generate data and validate it, returning a bundle of quality reports.

    Always runs integrity. Computes fidelity when ``reference_data`` is given
    and detection when ``detection`` is True (both require already-loaded
    reference rows — the caller loads them; the facade does no file IO here).
    Fidelity / detection lazy-import their ``[stats]``-extra deps so importing
    this module stays light (CLI startup-time contract).
    """
    result = generate(
        schema,
        config,
        seed=seed,
        default_rows=default_rows,
        engine=engine,
        reference_data=reference_data,
    )
    integrity = validate_integrity(result.tables_data, schema)

    fidelity: FidelityReport | None = None
    detection_report: DetectionReport | None = None
    if reference_data is not None:
        from dbsprout.quality.fidelity import validate_fidelity  # noqa: PLC0415

        fidelity = validate_fidelity(result.tables_data, reference_data, schema)
    if detection and reference_data is not None:
        from dbsprout.quality.detection import validate_detection  # noqa: PLC0415

        detection_report = validate_detection(result.tables_data, reference_data, schema, seed=seed)

    return ValidationOutcome(
        integrity=integrity,
        tables_data=result.tables_data,
        fidelity=fidelity,
        detection=detection_report,
    )
