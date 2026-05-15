"""Unit tests for dbsprout.train.serializer GReaT-style row serialization."""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import polars as pl
import pytest

from dbsprout.train.models import NullPolicy, SerializationResult
from dbsprout.train.serializer import (
    DataPreparer,
    _render_value,
    _shuffled_columns,
    serialize_row,
)

if TYPE_CHECKING:
    from pathlib import Path


# --- Task 2: _render_value type dispatch -----------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (34, "34"),
        (True, "true"),
        (False, "false"),
        (3.5, "3.5"),
        (Decimal("9.99"), "9.99"),
        ("John, Jr.", "John, Jr."),
        (uuid.UUID("12345678-1234-5678-1234-567812345678"), "12345678-1234-5678-1234-567812345678"),
        (datetime(2024, 1, 2, 3, 4, 5), "2024-01-02T03:04:05"),
        (date(2024, 1, 2), "2024-01-02"),
    ],
)
def test_render_value_types(value: object, expected: str) -> None:
    assert _render_value(value) == expected


def test_render_value_bytes_has_no_repr_prefix() -> None:
    out = _render_value(b"\x00\x01\xff")
    assert "b'" not in out
    assert "\\x" not in out
    assert out == "0001ff"


def test_render_value_fallback_uses_str() -> None:
    assert _render_value([1, 2]) == "[1, 2]"


# --- Task 3: serialize_row -------------------------------------------------


def test_serialize_row_fixed_order() -> None:
    out = serialize_row(
        {"id": 1, "name": "Ann"},
        table="users",
        column_order=["id", "name"],
        null_policy=NullPolicy.SKIP,
    )
    assert out == "[users] id is 1, name is Ann"


def test_serialize_row_skip_null() -> None:
    out = serialize_row(
        {"id": 1, "name": None},
        table="users",
        column_order=["id", "name"],
        null_policy=NullPolicy.SKIP,
    )
    assert out == "[users] id is 1"


def test_serialize_row_literal_null() -> None:
    out = serialize_row(
        {"id": 1, "name": None},
        table="users",
        column_order=["id", "name"],
        null_policy=NullPolicy.LITERAL,
    )
    assert out == "[users] id is 1, name is NULL"


def test_serialize_row_single_column_no_trailing_comma() -> None:
    out = serialize_row(
        {"id": 1},
        table="t",
        column_order=["id"],
        null_policy=NullPolicy.SKIP,
    )
    assert out == "[t] id is 1"


def test_serialize_row_all_null_skip_yields_bare_prefix() -> None:
    out = serialize_row(
        {"a": None},
        table="t",
        column_order=["a"],
        null_policy=NullPolicy.SKIP,
    )
    assert out == "[t]"


def test_serialize_row_value_with_comma_preserved() -> None:
    out = serialize_row(
        {"addr": "1, Main St"},
        table="t",
        column_order=["addr"],
        null_policy=NullPolicy.SKIP,
    )
    assert out == "[t] addr is 1, Main St"


# --- Task 4: seeded per-row shuffle ----------------------------------------


def test_shuffled_columns_is_deterministic() -> None:
    cols = ["a", "b", "c", "d", "e"]
    first = _shuffled_columns(cols, seed=7, table="users", row_index=3)
    second = _shuffled_columns(cols, seed=7, table="users", row_index=3)
    assert first == second
    assert sorted(first) == sorted(cols)


def test_shuffled_columns_does_not_mutate_input() -> None:
    cols = ["a", "b", "c"]
    _shuffled_columns(cols, seed=1, table="t", row_index=0)
    assert cols == ["a", "b", "c"]


def test_shuffled_columns_varies_by_seed() -> None:
    cols = [f"c{i}" for i in range(8)]
    a = _shuffled_columns(cols, seed=1, table="t", row_index=0)
    b = _shuffled_columns(cols, seed=2, table="t", row_index=0)
    assert a != b


# --- Task 5: DataPreparer.serialize in-memory API --------------------------


def test_dataparer_serialize_returns_json_lines() -> None:
    samples = {"users": pl.DataFrame({"id": [1, 2], "name": ["A", None]})}
    lines = DataPreparer().serialize(samples, seed=0, null_policy=NullPolicy.SKIP)
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert set(obj.keys()) == {"text", "table"}
        assert obj["table"] == "users"


def test_dataparer_serialize_honors_skip_policy() -> None:
    samples = {"users": pl.DataFrame({"id": [1], "name": [None]})}
    lines = DataPreparer().serialize(samples, seed=0, null_policy=NullPolicy.SKIP)
    obj = json.loads(lines[0])
    assert "name is" not in obj["text"]


def test_dataparer_serialize_honors_literal_policy() -> None:
    samples = {"users": pl.DataFrame({"id": [1], "name": [None]})}
    lines = DataPreparer().serialize(samples, seed=0, null_policy=NullPolicy.LITERAL)
    obj = json.loads(lines[0])
    assert "name is NULL" in obj["text"]


def test_dataparer_serialize_empty_frame_yields_no_lines() -> None:
    samples = {"users": pl.DataFrame({"id": []})}
    lines = DataPreparer().serialize(samples, seed=0, null_policy=NullPolicy.SKIP)
    assert lines == []


def test_dataparer_serialize_non_ascii_round_trips() -> None:
    samples = {"t": pl.DataFrame({"name": ["Renée Łódź 北京"]})}
    lines = DataPreparer().serialize(samples, seed=0, null_policy=NullPolicy.SKIP)
    obj = json.loads(lines[0])
    assert "Renée Łódź 北京" in obj["text"]


def test_dataparer_serialize_is_byte_identical_across_calls() -> None:
    samples = {
        "users": pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [True, False, True]})
    }
    a = DataPreparer().serialize(samples, seed=42, null_policy=NullPolicy.SKIP)
    b = DataPreparer().serialize(samples, seed=42, null_policy=NullPolicy.SKIP)
    assert a == b


def test_dataparer_serialize_sorts_tables() -> None:
    samples = {
        "zebra": pl.DataFrame({"id": [1]}),
        "alpha": pl.DataFrame({"id": [2]}),
    }
    lines = DataPreparer().serialize(samples, seed=0, null_policy=NullPolicy.SKIP)
    tables = [json.loads(line)["table"] for line in lines]
    assert tables == ["alpha", "zebra"]


# --- Task 6: prepare file-to-file ------------------------------------------


def _write_parquet(samples_dir: Path, name: str, df: pl.DataFrame) -> None:
    samples_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(samples_dir / f"{name}.parquet")


def test_prepare_writes_jsonl(tmp_path: Path) -> None:
    samples_dir = tmp_path / "samples"
    _write_parquet(samples_dir, "users", pl.DataFrame({"id": [1, 2], "name": ["A", "B"]}))
    _write_parquet(samples_dir, "orders", pl.DataFrame({"id": [9], "user_id": [1]}))
    out = tmp_path / "data.jsonl"

    result = DataPreparer().prepare(
        input_dir=tmp_path,
        output_path=out,
        seed=0,
        null_policy=NullPolicy.SKIP,
        quiet=True,
    )

    assert isinstance(result, SerializationResult)
    assert result.total_rows == 3
    text = out.read_text(encoding="utf-8")
    assert text.endswith("\n")
    lines = text.splitlines()
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)
        assert set(obj.keys()) == {"text", "table"}
    by_table = {t.table: t.rows_serialized for t in result.tables}
    assert by_table == {"users": 2, "orders": 1}


def test_prepare_counts_nulls_skipped(tmp_path: Path) -> None:
    samples_dir = tmp_path / "samples"
    _write_parquet(samples_dir, "users", pl.DataFrame({"id": [1, 2], "name": [None, None]}))
    out = tmp_path / "data.jsonl"

    result = DataPreparer().prepare(
        input_dir=tmp_path,
        output_path=out,
        seed=0,
        null_policy=NullPolicy.SKIP,
        quiet=True,
    )
    assert result.tables[0].nulls_skipped == 2


def test_prepare_empty_table_zero_rows(tmp_path: Path) -> None:
    samples_dir = tmp_path / "samples"
    _write_parquet(samples_dir, "empty", pl.DataFrame({"id": []}))
    out = tmp_path / "data.jsonl"

    result = DataPreparer().prepare(
        input_dir=tmp_path,
        output_path=out,
        seed=0,
        null_policy=NullPolicy.SKIP,
        quiet=True,
    )
    assert result.total_rows == 0
    assert result.tables[0].rows_serialized == 0
    assert out.read_text(encoding="utf-8") == ""


def test_prepare_missing_samples_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no Parquet sample files"):
        DataPreparer().prepare(
            input_dir=tmp_path,
            output_path=tmp_path / "data.jsonl",
            seed=0,
            null_policy=NullPolicy.SKIP,
            quiet=True,
        )


def test_prepare_byte_identical_across_runs(tmp_path: Path) -> None:
    samples_dir = tmp_path / "samples"
    _write_parquet(
        samples_dir,
        "users",
        pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.0, 2.0, 3.0]}),
    )
    out1 = tmp_path / "a.jsonl"
    out2 = tmp_path / "b.jsonl"
    DataPreparer().prepare(
        input_dir=tmp_path, output_path=out1, seed=11, null_policy=NullPolicy.SKIP, quiet=True
    )
    DataPreparer().prepare(
        input_dir=tmp_path, output_path=out2, seed=11, null_policy=NullPolicy.SKIP, quiet=True
    )
    assert out1.read_bytes() == out2.read_bytes()


def test_prepare_progress_not_quiet(tmp_path: Path) -> None:
    samples_dir = tmp_path / "samples"
    _write_parquet(samples_dir, "users", pl.DataFrame({"id": [1]}))
    out = tmp_path / "data.jsonl"
    result = DataPreparer().prepare(
        input_dir=tmp_path,
        output_path=out,
        seed=0,
        null_policy=NullPolicy.SKIP,
        quiet=False,
    )
    assert result.total_rows == 1
