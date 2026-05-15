"""GReaT-style row serialization: render extracted samples as ``column is value`` JSONL.

Implements the textual encoding from Borisov et al. 2022 ("Language Models are
Realistic Tabular Data Generators"): each row becomes a sentence
``[<table>] <col_a> is <v_a>, <col_b> is <v_b>, ...`` with the column order
**randomly permuted per row** so a causal LM does not learn positional bias.

Input is the per-table Parquet produced by :mod:`dbsprout.train.extractor`
(S-062); output is a single JSONL training corpus where every line is
``{"text": "<sentence>", "table": "<table>"}``.
"""

from __future__ import annotations

import json
import random
import time
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from dbsprout.train.models import (
    NullPolicy,
    SerializationResult,
    TableSerializationResult,
)

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

# 32-bit mask: keeps the per-row PRNG seed within a portable integer range.
_SEED_MASK = 0xFFFFFFFF


def _render_value(value: Any) -> str:
    """Render a single cell as plain text for a GReaT sentence.

    ``bool`` is checked before ``int`` (``bool`` is an ``int`` subclass).
    ``datetime``/``date`` use ISO 8601, ``bytes`` becomes lowercase hex (no
    ``b'...'`` repr noise), and everything else falls back to ``str()``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    # int / float / Decimal / UUID / str and any other type all render
    # via str(); the explicit branches above only cover types whose str()
    # would be lossy or noisy.
    return str(value)


def _shuffled_columns(
    columns: list[str],
    *,
    seed: int,
    table: str,
    row_index: int,
) -> list[str]:
    """Return a per-row permutation of *columns* without mutating the input.

    Seeds :class:`random.Random` with the stable string
    ``f"{seed}:{table}:{row_index}"``. CPython's ``Random`` hashes a ``str``
    seed with SHA-512 (random protocol v2), independent of ``PYTHONHASHSEED``
    — unlike the builtin ``hash()`` of a tuple — so two runs (even in
    separate processes) produce byte-identical permutations (AC-3).
    """
    ordered = list(columns)
    # A stable string seed: random.Random hashes str seeds with SHA-512
    # (CPython random v2), which is independent of PYTHONHASHSEED — unlike
    # the builtin hash() of a tuple. This guarantees byte-identical
    # permutations across separate processes (AC-3).
    rng = random.Random(f"{seed & _SEED_MASK}:{table}:{row_index}")  # noqa: S311
    rng.shuffle(ordered)
    return ordered


def serialize_row(
    row: dict[str, Any],
    *,
    table: str,
    column_order: list[str],
    null_policy: NullPolicy,
) -> str:
    """Render one row as a GReaT sentence ``[<table>] col is value, ...``.

    NULL cells follow *null_policy*: ``SKIP`` drops the clause entirely;
    ``LITERAL`` renders ``col is NULL``. A row with no surviving clauses
    serializes to the bare ``[<table>]`` prefix.
    """
    clauses: list[str] = []
    for col in column_order:
        value = row.get(col)
        if value is None:
            if null_policy is NullPolicy.LITERAL:
                clauses.append(f"{col} is NULL")
            continue
        clauses.append(f"{col} is {_render_value(value)}")
    if not clauses:
        return f"[{table}]"
    return f"[{table}] " + ", ".join(clauses)


class DataPreparer:
    """Built-in ``great`` train serializer.

    Reads S-062's per-table Parquet sample, renders each row as a GReaT-style
    sentence with a seeded per-row column shuffle, and writes a single JSONL
    training corpus.
    """

    def serialize(
        self,
        samples: dict[str, pl.DataFrame],
        *,
        seed: int,
        null_policy: NullPolicy,
    ) -> list[str]:
        """Serialize an in-memory ``table -> DataFrame`` map to JSONL lines.

        Tables are processed in sorted order so output is deterministic
        regardless of dict insertion order. Each returned element is a
        complete JSON string ``{"text": ..., "table": ...}``.
        """
        lines: list[str] = []
        for table in sorted(samples):
            df = samples[table]
            columns = list(df.columns)
            for row_index, row in enumerate(df.iter_rows(named=True)):
                order = _shuffled_columns(columns, seed=seed, table=table, row_index=row_index)
                sentence = serialize_row(
                    row, table=table, column_order=order, null_policy=null_policy
                )
                lines.append(json.dumps({"text": sentence, "table": table}, ensure_ascii=False))
        return lines

    def prepare(
        self,
        *,
        input_dir: Path,
        output_path: Path,
        seed: int,
        null_policy: NullPolicy,
        quiet: bool,
    ) -> SerializationResult:
        """Read ``<input_dir>/samples/*.parquet`` and write *output_path* JSONL.

        Raises :class:`FileNotFoundError` when no Parquet sample files exist
        (the CLI maps this to exit code 1). The output file always ends with a
        trailing newline; an all-empty corpus produces an empty file.
        """
        import polars as pl  # noqa: PLC0415 - lazy: keep polars off CLI startup path

        start = time.perf_counter()
        parquet_files = sorted((input_dir / "samples").glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"no Parquet sample files found under {input_dir / 'samples'}. "
                f"Run 'dbsprout train extract' first."
            )

        table_results: list[TableSerializationResult] = []
        all_lines: list[str] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            disable=quiet,
        ) as progress:
            overall = progress.add_task("Serializing", total=len(parquet_files))
            for path in parquet_files:
                table = path.stem
                df = pl.read_parquet(path)
                lines, nulls = self._serialize_table(table, df, seed=seed, null_policy=null_policy)
                all_lines.extend(lines)
                table_results.append(
                    TableSerializationResult(
                        table=table,
                        rows_serialized=len(lines),
                        nulls_skipped=nulls,
                    )
                )
                progress.update(overall, advance=1, description=f"Serializing {table}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(line + "\n" for line in all_lines)
        output_path.write_text(payload, encoding="utf-8")

        return SerializationResult(
            output_path=output_path,
            tables=tuple(table_results),
            total_rows=sum(r.rows_serialized for r in table_results),
            duration_seconds=time.perf_counter() - start,
        )

    def _serialize_table(
        self,
        table: str,
        df: pl.DataFrame,
        *,
        seed: int,
        null_policy: NullPolicy,
    ) -> tuple[list[str], int]:
        """Serialize one DataFrame, returning ``(json_lines, nulls_skipped)``."""
        columns = list(df.columns)
        lines: list[str] = []
        nulls_skipped = 0
        for row_index, row in enumerate(df.iter_rows(named=True)):
            if null_policy is NullPolicy.SKIP:
                nulls_skipped += sum(1 for v in row.values() if v is None)
            order = _shuffled_columns(columns, seed=seed, table=table, row_index=row_index)
            sentence = serialize_row(row, table=table, column_order=order, null_policy=null_policy)
            lines.append(json.dumps({"text": sentence, "table": table}, ensure_ascii=False))
        return lines, nulls_skipped
