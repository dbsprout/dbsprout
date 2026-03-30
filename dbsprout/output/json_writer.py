"""JSON/JSONL output writer.

Writes generated data as `.json` (pretty-printed array) or `.jsonl`
(one object per line) files with custom encoding for non-JSON-native types.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import date, datetime, time
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.schema.models import DatabaseSchema


class _SeedDataEncoder(json.JSONEncoder):
    """Custom JSON encoder for seed data types."""

    def default(self, o: Any) -> Any:
        if isinstance(o, (datetime, date, time)):
            return o.isoformat()
        if isinstance(o, uuid.UUID):
            return str(o)
        if isinstance(o, bytes):
            return o.hex()
        if isinstance(o, Decimal):
            return None if (o.is_nan() or o.is_infinite()) else float(o)
        if isinstance(o, (set, frozenset)):
            return list(o)
        return super().default(o)


def _sanitize_nan(value: Any) -> Any:
    """Convert float NaN/Inf to None for valid JSON."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _order_row(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """Reorder row keys to match schema column order."""
    return {col: _sanitize_nan(row.get(col)) for col in columns}


class JSONWriter:
    """Write generated data as JSON or JSONL files."""

    def write(
        self,
        tables_data: dict[str, list[dict[str, Any]]],
        schema: DatabaseSchema,
        insertion_order: list[str],
        output_dir: Path,
        fmt: Literal["json", "jsonl"] = "json",
    ) -> list[Path]:
        """Write JSON/JSONL files for each table.

        Returns list of written file paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        for idx, table_name in enumerate(insertion_order):
            rows = tables_data.get(table_name, [])
            if not rows:
                continue

            table_schema = schema.get_table(table_name)
            columns = (
                [col.name for col in table_schema.columns] if table_schema else list(rows[0].keys())
            )

            ext = "jsonl" if fmt == "jsonl" else "json"
            filename = f"{idx + 1:03d}_{table_name}.{ext}"
            filepath = output_dir / filename

            ordered_rows = [_order_row(row, columns) for row in rows]

            if fmt == "jsonl":
                _write_jsonl(filepath, ordered_rows)
            else:
                _write_json(filepath, ordered_rows)

            written.append(filepath)

        return written


def _write_json(filepath: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as a pretty-printed JSON array."""
    content = json.dumps(rows, cls=_SeedDataEncoder, indent=2, ensure_ascii=False)
    filepath.write_text(content + "\n", encoding="utf-8")


def _write_jsonl(filepath: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as JSONL (one JSON object per line)."""
    lines = [json.dumps(row, cls=_SeedDataEncoder, ensure_ascii=False) for row in rows]
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
