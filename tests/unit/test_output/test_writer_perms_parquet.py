"""File-permission hardening for parquet + mysql temp files (S-094)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

pl = pytest.importorskip("polars")

from dbsprout.output.mysql_load_data import _write_temp_file  # noqa: E402
from dbsprout.output.parquet_writer import ParquetWriter  # noqa: E402
from dbsprout.schema.models import (  # noqa: E402
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX only")


def test_parquet_writer_restricts_perms(tmp_path: Path) -> None:
    col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True)
    tbl = TableSchema(name="t", columns=[col], primary_key=["id"])
    schema = DatabaseSchema(tables=[tbl], dialect="postgresql")
    files = ParquetWriter().write({"t": [{"id": 1}]}, schema, ["t"], tmp_path)
    assert (files[0].stat().st_mode & 0o777) == 0o640


def test_mysql_temp_file_restricts_perms() -> None:
    path = _write_temp_file("a\tb\n")
    try:
        assert (os.stat(path).st_mode & 0o777) == 0o640
    finally:
        os.unlink(path)
