"""Tests for dbsprout.output.json_writer — JSON/JSONL output writer."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from dbsprout.output.json_writer import JSONWriter
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


# ── JSON format tests ───────────────────────────────────────────────


class TestJSONFormat:
    def test_json_array_of_objects(self, tmp_path: Path) -> None:
        """JSON output must be a valid array of objects."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [
                {"id": 1, "email": "alice@example.com"},
                {"id": 2, "email": "bob@example.com"},
            ],
        }
        writer = JSONWriter()
        paths = writer.write(data, schema, ["users"], tmp_path)

        content = paths[0].read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["id"] == 1
        assert parsed[1]["email"] == "bob@example.com"

    def test_json_pretty_printed(self, tmp_path: Path) -> None:
        """JSON output must be pretty-printed with 2-space indent."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}],
        }
        writer = JSONWriter()
        paths = writer.write(data, schema, ["users"], tmp_path)

        content = paths[0].read_text(encoding="utf-8")
        # Pretty-printed JSON has newlines and indentation
        assert "\n" in content
        assert "  " in content  # 2-space indent


# ── JSONL format tests ──────────────────────────────────────────────


class TestJSONLFormat:
    def test_jsonl_one_object_per_line(self, tmp_path: Path) -> None:
        """Each line in JSONL must be a valid JSON object."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [
                {"id": 1, "email": "alice@example.com"},
                {"id": 2, "email": "bob@example.com"},
            ],
        }
        writer = JSONWriter()
        paths = writer.write(data, schema, ["users"], tmp_path, fmt="jsonl")

        content = paths[0].read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert isinstance(obj, dict)
            assert "id" in obj
            assert "email" in obj

    def test_jsonl_extension(self, tmp_path: Path) -> None:
        """JSONL files should have .jsonl extension."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}],
        }
        writer = JSONWriter()
        paths = writer.write(data, schema, ["users"], tmp_path, fmt="jsonl")

        assert paths[0].name == "001_users.jsonl"


# ── File management tests ───────────────────────────────────────────


class TestJSONWriterFiles:
    def test_creates_files_with_prefix(self, tmp_path: Path) -> None:
        """Files should be named 001_users.json, 002_orders.json."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}],
            "orders": [{"id": 1, "user_id": 1}],
        }
        writer = JSONWriter()
        paths = writer.write(data, schema, ["users", "orders"], tmp_path)

        assert len(paths) == 2
        assert paths[0].name == "001_users.json"
        assert paths[1].name == "002_orders.json"

    def test_empty_table_skipped(self, tmp_path: Path) -> None:
        """Tables with 0 rows should not produce a file."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [],
            "orders": [{"id": 1, "user_id": 1}],
        }
        writer = JSONWriter()
        paths = writer.write(data, schema, ["users", "orders"], tmp_path)

        assert len(paths) == 1
        assert paths[0].name == "002_orders.json"

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        """Should create output directory if it doesn't exist."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": "a@b.com"}],
        }
        out = tmp_path / "nested" / "seeds"
        writer = JSONWriter()
        paths = writer.write(data, schema, ["users"], out)

        assert out.exists()
        assert len(paths) == 1


# ── Type formatting tests ───────────────────────────────────────────


class TestJSONTypeFormatting:
    def test_null_as_json_null(self, tmp_path: Path) -> None:
        """None values should be JSON null."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": None}],
        }
        writer = JSONWriter()
        paths = writer.write(data, schema, ["users"], tmp_path)

        parsed = json.loads(paths[0].read_text(encoding="utf-8"))
        assert parsed[0]["email"] is None

    def test_datetime_iso_8601(self, tmp_path: Path) -> None:
        """Datetime should be serialized as ISO 8601 string."""
        schema = _simple_schema()
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": dt}],
        }
        writer = JSONWriter()
        paths = writer.write(data, schema, ["users"], tmp_path)

        parsed = json.loads(paths[0].read_text(encoding="utf-8"))
        assert "2024-01-15" in parsed[0]["email"]
        assert "10:30:00" in parsed[0]["email"]

    def test_uuid_as_string(self, tmp_path: Path) -> None:
        """UUID should be serialized as string."""
        schema = _simple_schema()
        u = uuid.UUID("12345678-1234-5678-1234-567812345678")
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": u}],
        }
        writer = JSONWriter()
        paths = writer.write(data, schema, ["users"], tmp_path)

        parsed = json.loads(paths[0].read_text(encoding="utf-8"))
        assert parsed[0]["email"] == "12345678-1234-5678-1234-567812345678"

    def test_bytes_as_hex(self, tmp_path: Path) -> None:
        """Bytes should be serialized as hex string."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": b"\xde\xad\xbe\xef"}],
        }
        writer = JSONWriter()
        paths = writer.write(data, schema, ["users"], tmp_path)

        parsed = json.loads(paths[0].read_text(encoding="utf-8"))
        assert parsed[0]["email"] == "deadbeef"

    def test_decimal_as_float(self, tmp_path: Path) -> None:
        """Decimal should be serialized as a JSON number."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": Decimal("3.14")}],
        }
        writer = JSONWriter()
        paths = writer.write(data, schema, ["users"], tmp_path)

        parsed = json.loads(paths[0].read_text(encoding="utf-8"))
        assert parsed[0]["email"] == 3.14

    def test_set_as_list(self, tmp_path: Path) -> None:
        """Set values should be serialized as a JSON array."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "email": {3, 1, 2}}],
        }
        writer = JSONWriter()
        paths = writer.write(data, schema, ["users"], tmp_path)

        parsed = json.loads(paths[0].read_text(encoding="utf-8"))
        assert sorted(parsed[0]["email"]) == [1, 2, 3]

    def test_nan_as_null(self, tmp_path: Path) -> None:
        """NaN/Inf should be serialized as null."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [
                {"id": 1, "email": float("nan")},
                {"id": 2, "email": float("inf")},
            ],
        }
        writer = JSONWriter()
        paths = writer.write(data, schema, ["users"], tmp_path)

        parsed = json.loads(paths[0].read_text(encoding="utf-8"))
        assert parsed[0]["email"] is None
        assert parsed[1]["email"] is None

    def test_decimal_nan_as_null(self, tmp_path: Path) -> None:
        """Decimal NaN/Inf should be serialized as null."""
        schema = _simple_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "users": [
                {"id": 1, "email": Decimal("NaN")},
                {"id": 2, "email": Decimal("Infinity")},
            ],
        }
        writer = JSONWriter()
        paths = writer.write(data, schema, ["users"], tmp_path)

        parsed = json.loads(paths[0].read_text(encoding="utf-8"))
        assert parsed[0]["email"] is None
        assert parsed[1]["email"] is None


# ── Column ordering test ────────────────────────────────────────────


class TestColumnOrder:
    def test_column_order_matches_schema(self, tmp_path: Path) -> None:
        """JSON keys should follow schema column order."""
        schema = _simple_schema()
        # Pass row with keys in reverse order
        data: dict[str, list[dict[str, Any]]] = {
            "users": [{"email": "a@b.com", "id": 1}],
        }
        writer = JSONWriter()
        paths = writer.write(data, schema, ["users"], tmp_path)

        content = paths[0].read_text(encoding="utf-8")
        # id should appear before email in the output
        id_pos = content.index('"id"')
        email_pos = content.index('"email"')
        assert id_pos < email_pos
