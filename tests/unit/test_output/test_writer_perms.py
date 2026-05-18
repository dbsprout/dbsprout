"""File-permission hardening for sql/csv/json writers (S-094)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from dbsprout.output.csv_writer import CSVWriter
from dbsprout.output.json_writer import JSONWriter
from dbsprout.output.sql_writer import SQLWriter
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX only")


def _schema() -> DatabaseSchema:
    col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True)
    tbl = TableSchema(name="t", columns=[col], primary_key=["id"])
    return DatabaseSchema(tables=[tbl], dialect="postgresql")


def test_sql_writer_restricts_perms(tmp_path: Path) -> None:
    files = SQLWriter().write({"t": [{"id": 1}]}, _schema(), ["t"], tmp_path)
    assert (files[0].stat().st_mode & 0o777) == 0o640


def test_csv_writer_restricts_perms(tmp_path: Path) -> None:
    files = CSVWriter().write({"t": [{"id": 1}]}, _schema(), ["t"], tmp_path)
    assert (files[0].stat().st_mode & 0o777) == 0o640


def test_json_writer_restricts_perms(tmp_path: Path) -> None:
    files = JSONWriter().write({"t": [{"id": 1}]}, _schema(), ["t"], tmp_path)
    assert (files[0].stat().st_mode & 0o777) == 0o640
