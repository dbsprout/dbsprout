"""CSV output writer — RFC 4180 compliant.

Writes generated data as `.csv` files with proper quoting,
UTF-8 encoding, and consistent LF line endings.
"""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.schema.models import DatabaseSchema


def _format_csv_value(value: Any) -> str:
    """Format a Python value for CSV output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return str(value)


class CSVWriter:
    """Write generated data as CSV files."""

    def write(
        self,
        tables_data: dict[str, list[dict[str, Any]]],
        schema: DatabaseSchema,
        insertion_order: list[str],
        output_dir: Path,
    ) -> list[Path]:
        """Write CSV files for each table.

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

            filename = f"{idx + 1:03d}_{table_name}.csv"
            filepath = output_dir / filename

            with filepath.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                writer.writerow(columns)
                for row in rows:
                    writer.writerow(_format_csv_value(row.get(col)) for col in columns)

            written.append(filepath)

        return written
