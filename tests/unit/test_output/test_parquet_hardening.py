"""Parquet writer hardening: error-scrub, String dtype, tz, Snappy (S-094)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pl = pytest.importorskip("polars", reason="polars not installed ([data] extra)")

from dbsprout.output.parquet_writer import ParquetWriter  # noqa: E402
from dbsprout.schema.models import (  # noqa: E402
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path


def _schema(col_name: str, col_type: ColumnType) -> DatabaseSchema:
    col = ColumnSchema(name=col_name, data_type=col_type)
    tbl = TableSchema(name="t", columns=[col], primary_key=[])
    return DatabaseSchema(tables=[tbl], dialect="postgresql")


def test_parquet_type_mismatch_does_not_leak_values(tmp_path: Path) -> None:
    schema = _schema("amount", ColumnType.SMALLINT)
    secret = 99_999_999
    with pytest.raises(ValueError, match=r"Parquet column") as exc:
        ParquetWriter().write({"t": [{"amount": secret}]}, schema, ["t"], tmp_path)
    msg = str(exc.value)
    assert "amount" in msg
    assert "SMALLINT" in msg or "Int16" in msg
    assert str(secret) not in msg
