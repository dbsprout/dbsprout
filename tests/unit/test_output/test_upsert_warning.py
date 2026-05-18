"""Warn on --upsert with non-SQL output format (S-094 / S-045 review)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dbsprout.cli.commands.generate import _write_output
from dbsprout.generate.orchestrator import GenerateResult
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path


def _result_and_schema() -> tuple[GenerateResult, DatabaseSchema]:
    col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True)
    tbl = TableSchema(name="t", columns=[col], primary_key=["id"])
    schema = DatabaseSchema(tables=[tbl], dialect="postgresql")
    result = GenerateResult(tables_data={"t": [{"id": 1}]}, insertion_order=["t"])
    return result, schema


def test_upsert_with_csv_format_warns(capsys, tmp_path: Path) -> None:
    result, schema = _result_and_schema()
    _write_output(result, schema, ["t"], tmp_path, "csv", "postgresql", upsert=True)
    out = capsys.readouterr().out.lower()
    assert "upsert" in out
    assert "sql" in out


def test_upsert_with_sql_format_does_not_warn(capsys, tmp_path: Path) -> None:
    result, schema = _result_and_schema()
    _write_output(result, schema, ["t"], tmp_path, "sql", "postgresql", upsert=True)
    out = capsys.readouterr().out.lower()
    assert "only applies to" not in out


def test_no_upsert_no_warning(capsys, tmp_path: Path) -> None:
    result, schema = _result_and_schema()
    _write_output(result, schema, ["t"], tmp_path, "csv", "postgresql", upsert=False)
    out = capsys.readouterr().out.lower()
    assert "upsert" not in out
