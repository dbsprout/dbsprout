"""Tests for dbsprout.generate.engines.heuristic — heuristic generation engine."""

from __future__ import annotations

import re
import uuid
from datetime import date, time

from dbsprout.generate.engines.heuristic import (
    HeuristicEngine,
    _gen_random_bytes,
    _gen_random_date,
    _gen_random_decimal,
    _gen_random_float,
    _gen_random_json,
    _gen_random_list,
    _gen_random_text,
    _gen_random_time,
    _gen_ssn,
)
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)
from dbsprout.spec.heuristics import map_columns
from dbsprout.spec.models import GeneratorMapping


def _col(name: str, dtype: ColumnType = ColumnType.VARCHAR, **kw: object) -> ColumnSchema:
    return ColumnSchema(name=name, data_type=dtype, **kw)  # type: ignore[arg-type]


def _table(name: str, columns: list[ColumnSchema], **kw: object) -> TableSchema:
    return TableSchema(name=name, columns=columns, **kw)  # type: ignore[arg-type]


# ── Basic generation ─────────────────────────────────────────────────────


class TestBasicGeneration:
    def test_returns_correct_row_count(self) -> None:
        table = _table("users", [_col("id", ColumnType.INTEGER), _col("name")])
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine()
        rows = engine.generate_table(table, mappings["users"], 10)
        assert len(rows) == 10

    def test_row_has_all_columns(self) -> None:
        table = _table(
            "users",
            [
                _col("id", ColumnType.INTEGER),
                _col("email"),
                _col("name"),
            ],
        )
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine()
        rows = engine.generate_table(table, mappings["users"], 5)
        for row in rows:
            assert set(row.keys()) == {"id", "email", "name"}

    def test_email_contains_at(self) -> None:
        table = _table("users", [_col("email")])
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine()
        rows = engine.generate_table(table, mappings["users"], 5)
        for row in rows:
            assert "@" in str(row["email"])


# ── Type fallbacks ───────────────────────────────────────────────────────


class TestTypeFallbacks:
    def test_integer_fallback(self) -> None:
        table = _table("t", [_col("xyzzy", ColumnType.INTEGER)])
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine()
        rows = engine.generate_table(table, mappings["t"], 5)
        for row in rows:
            assert isinstance(row["xyzzy"], int)

    def test_boolean_fallback(self) -> None:
        table = _table("t", [_col("flag", ColumnType.BOOLEAN)])
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine()
        rows = engine.generate_table(table, mappings["t"], 5)
        for row in rows:
            assert isinstance(row["flag"], bool)

    def test_uuid_column(self) -> None:
        table = _table("t", [_col("qwxyz", ColumnType.UUID)])
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine()
        rows = engine.generate_table(table, mappings["t"], 3)
        for row in rows:
            uuid.UUID(str(row["qwxyz"]))  # validates UUID format


# ── FK and PK skip ───────────────────────────────────────────────────────


class TestSkipColumns:
    def test_fk_column_is_none(self) -> None:
        table = TableSchema(
            name="orders",
            columns=[
                _col("id", ColumnType.INTEGER, primary_key=True, autoincrement=True),
                _col("user_id", ColumnType.INTEGER),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"]),
            ],
        )
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine()
        rows = engine.generate_table(table, mappings["orders"], 3)
        for row in rows:
            assert row["user_id"] is None

    def test_autoincrement_pk_is_none(self) -> None:
        table = _table(
            "t",
            [
                _col("id", ColumnType.INTEGER, primary_key=True, autoincrement=True),
                _col("name"),
            ],
            primary_key=["id"],
        )
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine()
        rows = engine.generate_table(table, mappings["t"], 3)
        for row in rows:
            assert row["id"] is None


# ── Params respected ─────────────────────────────────────────────────────


class TestParamsRespected:
    def test_enum_values(self) -> None:
        table = _table("t", [_col("qwxyz", ColumnType.ENUM, enum_values=["a", "b", "c"])])
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine()
        rows = engine.generate_table(table, mappings["t"], 20)
        for row in rows:
            assert row["qwxyz"] in {"a", "b", "c"}

    def test_max_length_respected(self) -> None:
        table = _table("t", [_col("code", ColumnType.VARCHAR, max_length=5)])
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine()
        rows = engine.generate_table(table, mappings["t"], 20)
        for row in rows:
            assert len(str(row["code"])) <= 5


# ── Locale ───────────────────────────────────────────────────────────────


class TestMoreGenerators:
    def test_datetime_column(self) -> None:
        table = _table("t", [_col("created_at", ColumnType.TIMESTAMP)])
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine()
        rows = engine.generate_table(table, mappings["t"], 3)
        for row in rows:
            assert row["created_at"] is not None

    def test_text_column(self) -> None:
        table = _table("t", [_col("body", ColumnType.TEXT)])
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine()
        rows = engine.generate_table(table, mappings["t"], 3)
        for row in rows:
            assert isinstance(row["body"], str)
            assert len(row["body"]) > 0

    def test_decimal_column(self) -> None:
        table = _table("t", [_col("amount", ColumnType.DECIMAL, precision=10, scale=2)])
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine()
        rows = engine.generate_table(table, mappings["t"], 3)
        for row in rows:
            assert isinstance(row["amount"], (int, float))

    def test_first_name_column(self) -> None:
        table = _table("t", [_col("first_name")])
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine()
        rows = engine.generate_table(table, mappings["t"], 3)
        for row in rows:
            assert isinstance(row["first_name"], str)
            assert len(row["first_name"]) > 0

    def test_city_column(self) -> None:
        table = _table("t", [_col("city")])
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine()
        rows = engine.generate_table(table, mappings["t"], 3)
        for row in rows:
            assert isinstance(row["city"], str)

    def test_phone_column(self) -> None:
        table = _table("t", [_col("phone")])
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine()
        rows = engine.generate_table(table, mappings["t"], 3)
        for row in rows:
            assert row["phone"] is not None


class TestBuiltinGeneratorsDirect:
    """Test builtin generators directly for coverage."""

    def test_random_float(self) -> None:
        v = _gen_random_float({"min": 0.0, "max": 100.0})
        assert isinstance(v, float)

    def test_random_decimal(self) -> None:
        v = _gen_random_decimal({"precision": 8, "scale": 2})
        assert isinstance(v, float)

    def test_random_text(self) -> None:
        v = _gen_random_text({})
        assert isinstance(v, str)
        assert len(v) > 0

    def test_random_date(self) -> None:
        v = _gen_random_date({})
        assert isinstance(v, date)

    def test_random_time(self) -> None:
        v = _gen_random_time({})
        assert isinstance(v, time)

    def test_random_bytes(self) -> None:
        v = _gen_random_bytes({})
        assert isinstance(v, bytes)

    def test_random_json(self) -> None:
        v = _gen_random_json({})
        assert isinstance(v, dict)

    def test_random_list(self) -> None:
        v = _gen_random_list({})
        assert isinstance(v, list)

    def test_unknown_generator_fallback(self) -> None:
        """Unknown generator name falls back to random_string."""
        table = _table("t", [_col("weird", ColumnType.VARCHAR)])
        mapping = GeneratorMapping(
            generator_name="totally_unknown_gen", provider="builtin", confidence=0.5
        )
        engine = HeuristicEngine()
        rows = engine.generate_table(table, {"weird": mapping}, 3)
        for row in rows:
            assert isinstance(row["weird"], str)


class TestEdgeCases:
    def test_ssn_format(self) -> None:
        v = _gen_ssn()
        assert re.fullmatch(r"\d{3}-\d{2}-\d{4}", v)

    def test_zero_rows(self) -> None:
        table = _table("t", [_col("id", ColumnType.INTEGER)])
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine()
        rows = engine.generate_table(table, mappings["t"], 0)
        assert rows == []

    def test_none_mapping_returns_none(self) -> None:
        """Column with no mapping in dict → None values."""
        table = _table("t", [_col("weird", ColumnType.VARCHAR)])
        engine = HeuristicEngine()
        rows = engine.generate_table(table, {}, 3)  # empty mappings
        for row in rows:
            assert row["weird"] is None


class TestLocale:
    def test_locale_de(self) -> None:
        table = _table("t", [_col("email")])
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        engine = HeuristicEngine(locale="de")
        rows = engine.generate_table(table, mappings["t"], 3)
        assert len(rows) == 3  # just verify no crash
