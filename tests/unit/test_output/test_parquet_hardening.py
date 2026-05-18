"""Parquet writer hardening: error-scrub, String dtype, tz, Snappy (S-094)."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import pytest

pl = pytest.importorskip("polars", reason="polars not installed ([data] extra)")

from dbsprout.output.parquet_writer import (  # noqa: E402
    _COMPRESSION,
    ParquetWriter,
    _polars_dtype,
)
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


def test_parquet_uses_string_dtype_not_utf8() -> None:
    assert _polars_dtype(ColumnType.VARCHAR) == pl.String
    assert _polars_dtype(ColumnType.TEXT) == pl.String


def test_parquet_preserves_timezone(tmp_path: Path) -> None:
    schema = _schema("ts", ColumnType.TIMESTAMP)
    aware = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    files = ParquetWriter().write({"t": [{"ts": aware}]}, schema, ["t"], tmp_path)
    df = pl.read_parquet(files[0])
    assert df.schema["ts"].time_zone is not None


def test_parquet_naive_datetime_has_no_timezone(tmp_path: Path) -> None:
    schema = _schema("ts", ColumnType.TIMESTAMP)
    naive = dt.datetime(2026, 1, 1, 12, 0)
    files = ParquetWriter().write({"t": [{"ts": naive}]}, schema, ["t"], tmp_path)
    df = pl.read_parquet(files[0])
    assert df.schema["ts"].time_zone is None


def test_parquet_compression_is_snappy(tmp_path: Path) -> None:
    assert _COMPRESSION == "snappy"

    col = ColumnSchema(name="v", data_type=ColumnType.VARCHAR)
    tbl = TableSchema(name="t", columns=[col], primary_key=[])
    schema = DatabaseSchema(tables=[tbl], dialect="postgresql")
    rows = [{"v": "AAAAAAAAAAAAAAAAAAAA"} for _ in range(5_000)]
    files = ParquetWriter().write({"t": rows}, schema, ["t"], tmp_path)

    # Round-trips and is compressed (snappy shrinks 5k identical strings well
    # below the ~100KB uncompressed size).
    df = pl.read_parquet(files[0])
    assert df.height == 5_000
    assert files[0].stat().st_size < 5_000
