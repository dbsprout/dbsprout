"""Tests for dbsprout.output.csv_writer — CSV output writer."""

from __future__ import annotations

import csv
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from dbsprout.output.csv_writer import CSVWriter
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


def _read_csv(path: Path) -> list[list[str]]:
    """Read a CSV file and return all rows."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


class TestCSVWriterFiles:
    def test_creates_csv_files_with_prefix(self, tmp_path: Path) -> None:
        """Files should be named 001_users.csv, 002_orders.csv."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}],
            "orders": [{"id": 1, "user_id": 1}],
        }
        writer = CSVWriter()
        paths = writer.write(data, schema, ["users", "orders"], tmp_path)

        assert len(paths) == 2
        assert paths[0].name == "001_users.csv"
        assert paths[1].name == "002_orders.csv"
        assert all(p.exists() for p in paths)

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        """Should create output directory if it doesn't exist."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}],
        }
        out = tmp_path / "nested" / "seeds"
        writer = CSVWriter()
        paths = writer.write(data, schema, ["users"], out)

        assert out.exists()
        assert len(paths) == 1

    def test_empty_table_skipped(self, tmp_path: Path) -> None:
        """Tables with 0 rows should not produce a file."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [],
            "orders": [{"id": 1, "user_id": 1}],
        }
        writer = CSVWriter()
        paths = writer.write(data, schema, ["users", "orders"], tmp_path)

        assert len(paths) == 1
        assert paths[0].name == "002_orders.csv"


class TestCSVHeaders:
    def test_csv_has_headers(self, tmp_path: Path) -> None:
        """First row must contain column names from schema."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "alice@example.com"}],
        }
        writer = CSVWriter()
        paths = writer.write(data, schema, ["users"], tmp_path)

        rows = _read_csv(paths[0])
        assert rows[0] == ["id", "email"]


class TestCSVValues:
    def test_csv_values_match_data(self, tmp_path: Path) -> None:
        """Data rows must match generated values."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [
                {"id": 1, "email": "alice@example.com"},
                {"id": 2, "email": "bob@example.com"},
            ],
        }
        writer = CSVWriter()
        paths = writer.write(data, schema, ["users"], tmp_path)

        rows = _read_csv(paths[0])
        assert len(rows) == 3  # header + 2 data rows
        assert rows[1] == ["1", "alice@example.com"]
        assert rows[2] == ["2", "bob@example.com"]

    def test_null_as_empty_string(self, tmp_path: Path) -> None:
        """None values should be written as empty strings."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": None}],
        }
        writer = CSVWriter()
        paths = writer.write(data, schema, ["users"], tmp_path)

        rows = _read_csv(paths[0])
        assert rows[1] == ["1", ""]

    def test_bool_lowercase(self, tmp_path: Path) -> None:
        """Booleans should be lowercase true/false."""
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
            "flags": [
                {"id": 1, "active": True},
                {"id": 2, "active": False},
            ],
        }
        writer = CSVWriter()
        paths = writer.write(data, schema, ["flags"], tmp_path)

        rows = _read_csv(paths[0])
        assert rows[1][1] == "true"
        assert rows[2][1] == "false"


class TestCSVSpecialChars:
    def test_special_chars_quoted(self, tmp_path: Path) -> None:
        """Commas, quotes, newlines in values must be properly escaped."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [
                {"id": 1, "email": "has,comma"},
                {"id": 2, "email": 'has"quote'},
                {"id": 3, "email": "has\nnewline"},
            ],
        }
        writer = CSVWriter()
        paths = writer.write(data, schema, ["users"], tmp_path)

        rows = _read_csv(paths[0])
        assert rows[1][1] == "has,comma"
        assert rows[2][1] == 'has"quote'
        assert rows[3][1] == "has\nnewline"


class TestCSVTypeFormatting:
    def test_datetime_iso(self, tmp_path: Path) -> None:
        """Datetime values should be ISO 8601 strings."""
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
        writer = CSVWriter()
        paths = writer.write(data, schema, ["events"], tmp_path)

        rows = _read_csv(paths[0])
        assert "2024-01-15" in rows[1][1]
        assert "10:30" in rows[1][1]

    def test_uuid_as_string(self, tmp_path: Path) -> None:
        """UUID values should be written as strings."""
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
        writer = CSVWriter()
        paths = writer.write(data, schema, ["tokens"], tmp_path)

        rows = _read_csv(paths[0])
        assert rows[1][0] == "12345678-1234-5678-1234-567812345678"

    def test_bytes_as_hex(self, tmp_path: Path) -> None:
        """Bytes values should be written as hex strings."""
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
        writer = CSVWriter()
        paths = writer.write(data, schema, ["blobs"], tmp_path)

        rows = _read_csv(paths[0])
        assert rows[1][1] == "deadbeef"

    def test_dict_as_json(self, tmp_path: Path) -> None:
        """Dict values should be written as JSON strings."""
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
        writer = CSVWriter()
        paths = writer.write(data, schema, ["configs"], tmp_path)

        rows = _read_csv(paths[0])
        assert rows[1][1] == '{"key": "value"}'


class TestCSVDeterministic:
    def test_same_data_same_output(self, tmp_path: Path) -> None:
        """Same input data must produce identical CSV files."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [
                {"id": 1, "email": "a@b.com"},
                {"id": 2, "email": "c@d.com"},
            ],
        }
        writer = CSVWriter()

        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        paths1 = writer.write(data, schema, ["users"], out1)
        paths2 = writer.write(data, schema, ["users"], out2)

        assert paths1[0].read_bytes() == paths2[0].read_bytes()
