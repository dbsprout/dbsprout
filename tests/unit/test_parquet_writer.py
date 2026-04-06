"""Tests for dbsprout.output.parquet_writer — Parquet output writer."""

from __future__ import annotations

import uuid
from datetime import date, datetime, time, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import polars as pl
import pytest

from dbsprout.output.parquet_writer import ParquetWriter
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path


def _simple_schema() -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="users",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                    ),
                    ColumnSchema(
                        name="email",
                        data_type=ColumnType.VARCHAR,
                        nullable=False,
                    ),
                ],
                primary_key=["id"],
            ),
            TableSchema(
                name="orders",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                    ),
                    ColumnSchema(
                        name="user_id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                    ),
                ],
                primary_key=["id"],
                foreign_keys=[
                    ForeignKeySchema(
                        columns=["user_id"],
                        ref_table="users",
                        ref_columns=["id"],
                    ),
                ],
            ),
        ],
        dialect="postgresql",
    )


class TestParquetWriterFiles:
    def test_creates_parquet_files_with_prefix(self, tmp_path: Path) -> None:
        """Files should be named 001_users.parquet, 002_orders.parquet."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}],
            "orders": [{"id": 1, "user_id": 1}],
        }
        paths = ParquetWriter().write(data, schema, ["users", "orders"], tmp_path)

        assert len(paths) == 2
        assert paths[0].name == "001_users.parquet"
        assert paths[1].name == "002_orders.parquet"
        assert all(p.exists() for p in paths)

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        """Should create output directory if it doesn't exist."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}],
        }
        out = tmp_path / "nested" / "seeds"
        paths = ParquetWriter().write(data, schema, ["users"], out)

        assert out.exists()
        assert len(paths) == 1

    def test_empty_table_writes_schema_only(self, tmp_path: Path) -> None:
        """Empty tables should produce a Parquet file with schema but zero rows."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [],
            "orders": [{"id": 1, "user_id": 1}],
        }
        paths = ParquetWriter().write(data, schema, ["users", "orders"], tmp_path)

        assert len(paths) == 2
        df = pl.read_parquet(paths[0])
        assert len(df) == 0
        assert "id" in df.columns
        assert "email" in df.columns


class TestParquetTypes:
    def test_integer_columns(self, tmp_path: Path) -> None:
        """Integer values should round-trip correctly."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 42, "email": "test@test.com"}],
        }
        paths = ParquetWriter().write(data, schema, ["users"], tmp_path)
        df = pl.read_parquet(paths[0])

        assert df["id"][0] == 42
        assert df["id"].dtype == pl.Int64

    def test_string_columns(self, tmp_path: Path) -> None:
        """String values should round-trip correctly."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "hello@world.com"}],
        }
        paths = ParquetWriter().write(data, schema, ["users"], tmp_path)
        df = pl.read_parquet(paths[0])

        assert df["email"][0] == "hello@world.com"
        assert df["email"].dtype == pl.Utf8

    def test_boolean_columns(self, tmp_path: Path) -> None:
        """Boolean values should round-trip correctly."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="flags",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="active",
                            data_type=ColumnType.BOOLEAN,
                            nullable=False,
                        ),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="postgresql",
        )
        data: dict[str, list[dict[str, Any]]] = {
            "flags": [{"id": 1, "active": True}, {"id": 2, "active": False}],
        }
        paths = ParquetWriter().write(data, schema, ["flags"], tmp_path)
        df = pl.read_parquet(paths[0])

        assert df["active"][0] is True
        assert df["active"][1] is False
        assert df["active"].dtype == pl.Boolean

    def test_datetime_columns(self, tmp_path: Path) -> None:
        """Datetime values should round-trip correctly."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="events",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="created_at",
                            data_type=ColumnType.TIMESTAMP,
                            nullable=False,
                        ),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="postgresql",
        )
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        data: dict[str, list[dict[str, Any]]] = {
            "events": [{"id": 1, "created_at": dt}],
        }
        paths = ParquetWriter().write(data, schema, ["events"], tmp_path)
        df = pl.read_parquet(paths[0])

        result = df["created_at"][0]
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_date_columns(self, tmp_path: Path) -> None:
        """Date values should round-trip correctly."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="events",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="event_date",
                            data_type=ColumnType.DATE,
                            nullable=False,
                        ),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="postgresql",
        )
        d = date(2024, 6, 15)
        data: dict[str, list[dict[str, Any]]] = {
            "events": [{"id": 1, "event_date": d}],
        }
        paths = ParquetWriter().write(data, schema, ["events"], tmp_path)
        df = pl.read_parquet(paths[0])

        assert df["event_date"][0] == d
        assert df["event_date"].dtype == pl.Date

    def test_uuid_as_string(self, tmp_path: Path) -> None:
        """UUID values should be stored as Utf8 strings."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="tokens",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.UUID,
                            nullable=False,
                            primary_key=True,
                        ),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="postgresql",
        )
        u = uuid.UUID("12345678-1234-5678-1234-567812345678")
        data: dict[str, list[dict[str, Any]]] = {
            "tokens": [{"id": u}],
        }
        paths = ParquetWriter().write(data, schema, ["tokens"], tmp_path)
        df = pl.read_parquet(paths[0])

        assert df["id"][0] == "12345678-1234-5678-1234-567812345678"
        assert df["id"].dtype == pl.Utf8

    def test_bytes_as_binary(self, tmp_path: Path) -> None:
        """Bytes values should be stored as Binary dtype."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="blobs",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="data",
                            data_type=ColumnType.BINARY,
                            nullable=True,
                        ),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="postgresql",
        )
        data: dict[str, list[dict[str, Any]]] = {
            "blobs": [{"id": 1, "data": b"\xde\xad\xbe\xef"}],
        }
        paths = ParquetWriter().write(data, schema, ["blobs"], tmp_path)
        df = pl.read_parquet(paths[0])

        assert df["data"][0] == b"\xde\xad\xbe\xef"
        assert df["data"].dtype == pl.Binary

    def test_json_as_string(self, tmp_path: Path) -> None:
        """Dict/list values should be stored as Utf8 JSON strings."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="configs",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="settings",
                            data_type=ColumnType.JSON,
                            nullable=True,
                        ),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="postgresql",
        )
        data: dict[str, list[dict[str, Any]]] = {
            "configs": [{"id": 1, "settings": {"key": "value"}}],
        }
        paths = ParquetWriter().write(data, schema, ["configs"], tmp_path)
        df = pl.read_parquet(paths[0])

        assert df["settings"][0] == '{"key": "value"}'
        assert df["settings"].dtype == pl.Utf8

    def test_float_columns(self, tmp_path: Path) -> None:
        """Float values should round-trip correctly."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="metrics",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="score",
                            data_type=ColumnType.FLOAT,
                            nullable=False,
                        ),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="postgresql",
        )
        data: dict[str, list[dict[str, Any]]] = {
            "metrics": [{"id": 1, "score": 3.14}],
        }
        paths = ParquetWriter().write(data, schema, ["metrics"], tmp_path)
        df = pl.read_parquet(paths[0])

        assert abs(df["score"][0] - 3.14) < 0.001
        assert df["score"].dtype == pl.Float64

    def test_time_columns(self, tmp_path: Path) -> None:
        """Time values should round-trip correctly."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="schedules",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="start_time",
                            data_type=ColumnType.TIME,
                            nullable=False,
                        ),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="postgresql",
        )
        t = time(14, 30, 0)
        data: dict[str, list[dict[str, Any]]] = {
            "schedules": [{"id": 1, "start_time": t}],
        }
        paths = ParquetWriter().write(data, schema, ["schedules"], tmp_path)
        df = pl.read_parquet(paths[0])

        assert df["start_time"][0] == t
        assert df["start_time"].dtype == pl.Time


class TestParquetNulls:
    def test_null_values(self, tmp_path: Path) -> None:
        """None values should become Parquet nulls."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": None}],
        }
        paths = ParquetWriter().write(data, schema, ["users"], tmp_path)
        df = pl.read_parquet(paths[0])

        assert df["email"][0] is None

    def test_nan_becomes_null(self, tmp_path: Path) -> None:
        """Float NaN should become null in Parquet."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="metrics",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="score",
                            data_type=ColumnType.FLOAT,
                            nullable=True,
                        ),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="postgresql",
        )
        data: dict[str, list[dict[str, Any]]] = {
            "metrics": [{"id": 1, "score": float("nan")}],
        }
        paths = ParquetWriter().write(data, schema, ["metrics"], tmp_path)
        df = pl.read_parquet(paths[0])

        assert df["score"][0] is None

    def test_inf_becomes_null(self, tmp_path: Path) -> None:
        """Float Inf should become null in Parquet."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="metrics",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="score",
                            data_type=ColumnType.FLOAT,
                            nullable=True,
                        ),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="postgresql",
        )
        data: dict[str, list[dict[str, Any]]] = {
            "metrics": [{"id": 1, "score": float("inf")}],
        }
        paths = ParquetWriter().write(data, schema, ["metrics"], tmp_path)
        df = pl.read_parquet(paths[0])

        assert df["score"][0] is None

    def test_decimal_value(self, tmp_path: Path) -> None:
        """Decimal values should be converted to float."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="metrics",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="score",
                            data_type=ColumnType.DECIMAL,
                            nullable=True,
                        ),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="postgresql",
        )
        data: dict[str, list[dict[str, Any]]] = {
            "metrics": [{"id": 1, "score": Decimal("3.14")}],
        }
        paths = ParquetWriter().write(data, schema, ["metrics"], tmp_path)
        df = pl.read_parquet(paths[0])

        assert abs(df["score"][0] - 3.14) < 0.001

    def test_decimal_nan_becomes_null(self, tmp_path: Path) -> None:
        """Decimal NaN should become null in Parquet."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="metrics",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="score",
                            data_type=ColumnType.DECIMAL,
                            nullable=True,
                        ),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="postgresql",
        )
        data: dict[str, list[dict[str, Any]]] = {
            "metrics": [{"id": 1, "score": Decimal("NaN")}],
        }
        paths = ParquetWriter().write(data, schema, ["metrics"], tmp_path)
        df = pl.read_parquet(paths[0])

        assert df["score"][0] is None


class TestParquetSpecial:
    def test_deterministic_output(self, tmp_path: Path) -> None:
        """Same input data must produce identical Parquet files."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [
                {"id": 1, "email": "a@b.com"},
                {"id": 2, "email": "c@d.com"},
            ],
        }
        writer = ParquetWriter()

        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        paths1 = writer.write(data, schema, ["users"], out1)
        paths2 = writer.write(data, schema, ["users"], out2)

        assert paths1[0].read_bytes() == paths2[0].read_bytes()


class TestParquetSecurity:
    def test_path_traversal_sanitized(self, tmp_path: Path) -> None:
        """Table names with path separators should be sanitized."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="../escape",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="postgresql",
        )
        data: dict[str, list[dict[str, Any]]] = {
            "../escape": [{"id": 1}],
        }
        paths = ParquetWriter().write(data, schema, ["../escape"], tmp_path)

        assert len(paths) == 1
        assert paths[0].resolve().is_relative_to(tmp_path.resolve())
        assert "/" not in paths[0].name.replace(".parquet", "")

    def test_circular_reference_returns_null(self, tmp_path: Path) -> None:
        """Circular dict references should not crash, returning null instead."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="configs",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="settings",
                            data_type=ColumnType.JSON,
                            nullable=True,
                        ),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="postgresql",
        )
        circular: dict[str, Any] = {}
        circular["self"] = circular
        data: dict[str, list[dict[str, Any]]] = {
            "configs": [{"id": 1, "settings": circular}],
        }
        paths = ParquetWriter().write(data, schema, ["configs"], tmp_path)
        df = pl.read_parquet(paths[0])

        assert df["settings"][0] is None

    def test_set_value_serialized(self, tmp_path: Path) -> None:
        """Set values should be JSON-serialized without crashing."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="tags",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="labels",
                            data_type=ColumnType.JSON,
                            nullable=True,
                        ),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="postgresql",
        )
        data: dict[str, list[dict[str, Any]]] = {
            "tags": [{"id": 1, "labels": {"a", "b", "c"}}],
        }
        paths = ParquetWriter().write(data, schema, ["tags"], tmp_path)
        df = pl.read_parquet(paths[0])

        assert df["labels"][0] == '["a", "b", "c"]'


class TestParquetImportGuard:
    def test_import_error_without_polars(self, tmp_path: Path) -> None:
        """Should raise ImportError with helpful message when polars is missing."""
        import importlib  # noqa: PLC0415

        import dbsprout.output.parquet_writer as mod  # noqa: PLC0415

        try:
            with patch.dict("sys.modules", {"polars": None}):
                importlib.reload(mod)

                with pytest.raises(ImportError, match="polars"):
                    mod.ParquetWriter().write({}, _simple_schema(), [], tmp_path)
        finally:
            importlib.reload(mod)
