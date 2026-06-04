"""Live-DB ``--output-format direct`` adapter for the ``generate`` command.

This is a CLI-resident output-target adapter, not pipeline orchestration: it
dispatches generated data to the fastest available per-dialect writer and owns
the CLI fallback UX (psycopg/pymysql ``ImportError`` handling, per-dialect
warnings). The core service facade in ``dbsprout/core/service.py`` deliberately
leaves this path in the CLI because its fallbacks are console-coupled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from rich.console import Console

from dbsprout.output.dialect import detect_direct_dialect as _detect_direct_dialect

if TYPE_CHECKING:
    from dbsprout.generate.orchestrator import GenerateResult
    from dbsprout.output.models import InsertResult
    from dbsprout.schema.models import DatabaseSchema

console = Console()

__all__ = ["_detect_direct_dialect", "_run_direct_insert"]


def _run_direct_insert(  # noqa: PLR0913
    result: GenerateResult,
    schema: DatabaseSchema,
    insertion_order: list[str],
    target_db: str,
    insert_method: str = "auto",
    upsert: bool = False,
) -> None:
    """Dispatch direct insertion to the appropriate dialect writer.

    Parameters
    ----------
    result:
        Generated data from the orchestrator.
    schema:
        Unified database schema.
    insertion_order:
        Table names in FK-safe insertion order.
    target_db:
        SQLAlchemy connection URL.
    insert_method:
        One of ``"auto"``, ``"copy"``, ``"load_data"``, ``"batch"``.
        ``"auto"`` selects the fastest available writer for the dialect.
    upsert:
        When True, the PG COPY and MySQL LOAD DATA writers perform an
        UPSERT via a staging table. The SQLAlchemy batch fallback does not
        support UPSERT; a one-time warning is printed in that case.
    """
    dialect = _detect_direct_dialect(target_db)

    # Validate method/dialect compatibility
    if insert_method == "copy" and dialect != "postgresql":
        console.print("[red]Error:[/red] --insert-method copy is only available for PostgreSQL.")
        raise typer.Exit(code=1)
    if insert_method == "load_data" and dialect != "mysql":
        console.print("[red]Error:[/red] --insert-method load_data is only available for MySQL.")
        raise typer.Exit(code=1)

    method_name: str
    insert_result: InsertResult

    if upsert and (insert_method == "batch" or dialect in ("sqlite", "mssql")):
        console.print(
            "[yellow]Warning:[/yellow] --upsert is not supported by the "
            "SQLAlchemy batch insert path; rows will be inserted without "
            "UPSERT semantics."
        )

    if insert_method == "batch":
        from dbsprout.output.sa_batch import SaBatchWriter  # noqa: PLC0415

        method_name = f"SQLAlchemy batch INSERT ({dialect})"
        insert_result = SaBatchWriter().write(
            result.tables_data, schema, insertion_order, target_db
        )

    elif insert_method == "copy" or (insert_method == "auto" and dialect == "postgresql"):
        try:
            import psycopg  # noqa: F401, PLC0415

            from dbsprout.output.pg_copy import PgCopyWriter  # noqa: PLC0415

            method_name = (
                "PostgreSQL COPY (auto-detected)"
                if insert_method == "auto"
                else "PostgreSQL COPY (user-selected)"
            )
            insert_result = PgCopyWriter().write(
                result.tables_data,
                schema,
                insertion_order,
                target_db,
                upsert=upsert,
            )
        except ImportError:
            console.print(
                "[yellow]Warning:[/yellow] psycopg not installed, falling back to batch INSERT."
            )
            if upsert:
                console.print(
                    "[yellow]Warning:[/yellow] --upsert is ignored on the batch INSERT fallback."
                )
            from dbsprout.output.sa_batch import SaBatchWriter  # noqa: PLC0415

            method_name = "SQLAlchemy batch INSERT (fallback)"
            insert_result = SaBatchWriter().write(
                result.tables_data, schema, insertion_order, target_db
            )

    elif insert_method == "load_data" or (insert_method == "auto" and dialect == "mysql"):
        try:
            import pymysql  # type: ignore[import-untyped]  # noqa: F401, PLC0415

            from dbsprout.output.mysql_load_data import (  # noqa: PLC0415
                MysqlLoadDataWriter,
            )

            method_name = (
                "MySQL LOAD DATA (auto-detected)"
                if insert_method == "auto"
                else "MySQL LOAD DATA (user-selected)"
            )
            insert_result = MysqlLoadDataWriter().write(
                result.tables_data,
                schema,
                insertion_order,
                target_db,
                upsert=upsert,
            )
        except ImportError:
            console.print(
                "[yellow]Warning:[/yellow] pymysql not installed, falling back to batch INSERT."
            )
            if upsert:
                console.print(
                    "[yellow]Warning:[/yellow] --upsert is ignored on the batch INSERT fallback."
                )
            from dbsprout.output.sa_batch import SaBatchWriter  # noqa: PLC0415

            method_name = "SQLAlchemy batch INSERT (fallback)"
            insert_result = SaBatchWriter().write(
                result.tables_data, schema, insertion_order, target_db
            )

    else:
        # sqlite, mssql, unknown -- all use SaBatchWriter
        from dbsprout.output.sa_batch import SaBatchWriter  # noqa: PLC0415

        method_name = f"SQLAlchemy batch INSERT ({dialect})"
        insert_result = SaBatchWriter().write(
            result.tables_data, schema, insertion_order, target_db
        )

    console.print(f"[blue]Insert method:[/blue] {method_name}")
    console.print(
        f"[green]Inserted {insert_result.total_rows} rows "
        f"into {insert_result.tables_inserted} tables "
        f"in {insert_result.duration_seconds:.3f}s[/green]"
    )
