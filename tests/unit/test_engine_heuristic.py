"""Tests for dbsprout.generate.engines.heuristic — heuristic generation engine."""

from __future__ import annotations

import random
import re
import uuid
from datetime import date, datetime, time

from dbsprout.generate.engines.heuristic import (
    HeuristicEngine,
    _gen_random_bool,
    _gen_random_bytes,
    _gen_random_choice,
    _gen_random_date,
    _gen_random_datetime,
    _gen_random_decimal,
    _gen_random_float,
    _gen_random_int,
    _gen_random_json,
    _gen_random_list,
    _gen_random_string,
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

    def setup_method(self) -> None:
        self._rng = random.Random(0)  # noqa: S311

    def test_random_float(self) -> None:
        v = _gen_random_float({"min": 0.0, "max": 100.0}, self._rng)
        assert isinstance(v, float)

    def test_random_decimal(self) -> None:
        v = _gen_random_decimal({"precision": 8, "scale": 2}, self._rng)
        assert isinstance(v, float)

    def test_random_text(self) -> None:
        v = _gen_random_text({}, self._rng)
        assert isinstance(v, str)
        assert len(v) > 0

    def test_random_date(self) -> None:
        v = _gen_random_date({}, self._rng)
        assert isinstance(v, date)

    def test_random_time(self) -> None:
        v = _gen_random_time({}, self._rng)
        assert isinstance(v, time)

    def test_random_bytes(self) -> None:
        v = _gen_random_bytes({}, self._rng)
        assert isinstance(v, bytes)

    def test_random_json(self) -> None:
        v = _gen_random_json({}, self._rng)
        assert isinstance(v, dict)

    def test_random_list(self) -> None:
        v = _gen_random_list({}, self._rng)
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
        v = _gen_ssn(random.Random(0))  # noqa: S311
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


# ── S-104: Per-column random.Random — global RNG isolation ───────────────


class TestGlobalRngNotMutated:
    """S-104 acceptance: generate_table must NOT mutate process-global random state."""

    def _make_table(self, name: str) -> tuple[TableSchema, dict[str, GeneratorMapping]]:
        table = _table(
            name,
            [
                _col("name"),
                _col("age", ColumnType.INTEGER),
                _col("score", ColumnType.FLOAT),
                _col("active", ColumnType.BOOLEAN),
                _col("notes", ColumnType.TEXT),
                _col("created_at", ColumnType.TIMESTAMP),
            ],
        )
        schema = DatabaseSchema(tables=[table])
        mappings = map_columns(schema)
        return table, mappings[name]

    def test_generate_table_does_not_mutate_global_rng(self) -> None:
        """Global random.Random state must be identical before and after generate_table."""
        table_a, mappings_a = self._make_table("table_a")
        table_b, mappings_b = self._make_table("table_b")
        engine = HeuristicEngine(seed=42)

        before_state = random.getstate()

        engine.generate_table(table_a, mappings_a, 20)
        engine.generate_table(table_b, mappings_b, 20)

        after_state = random.getstate()
        assert before_state == after_state, (
            "HeuristicEngine.generate_table mutated the global random module state. "
            "Use a per-column random.Random instance instead of random.seed()."
        )

    def test_global_rng_sequence_preserved(self) -> None:
        """Values from global random before generate_table appear in correct sequence after."""
        table, mappings = self._make_table("seq_test")
        engine = HeuristicEngine(seed=99)

        random.seed(1234)
        first_val = random.random()  # noqa: S311

        engine.generate_table(table, mappings, 10)

        second_val = random.random()  # noqa: S311

        random.seed(1234)
        expected_first = random.random()  # noqa: S311
        expected_second = random.random()  # noqa: S311

        assert first_val == expected_first
        assert second_val == expected_second, (
            "Global random sequence was disrupted by generate_table; "
            "engine must use a local random.Random instance."
        )


# ── S-104: Builtin generators accept rng parameter ───────────────────────


class TestBuiltinGeneratorsAcceptRng:
    """S-104: Every builtin generator must accept a random.Random instance."""

    def setup_method(self) -> None:
        self.rng = random.Random(42)  # noqa: S311

    def test_gen_random_int_with_rng(self) -> None:
        v = _gen_random_int({"min": 0, "max": 100}, self.rng)
        assert isinstance(v, int)
        assert 0 <= v <= 100

    def test_gen_random_float_with_rng(self) -> None:
        v = _gen_random_float({"min": 0.0, "max": 10.0}, self.rng)
        assert isinstance(v, float)
        assert 0.0 <= v <= 10.0

    def test_gen_random_decimal_with_rng(self) -> None:
        v = _gen_random_decimal({"precision": 8, "scale": 2}, self.rng)
        assert isinstance(v, float)

    def test_gen_random_bool_with_rng(self) -> None:
        v = _gen_random_bool({}, self.rng)
        assert isinstance(v, bool)

    def test_gen_random_string_with_rng(self) -> None:
        v = _gen_random_string({"max_length": 10}, self.rng)
        assert isinstance(v, str)
        assert len(v) <= 10

    def test_gen_random_text_with_rng(self) -> None:
        v = _gen_random_text({}, self.rng)
        assert isinstance(v, str)
        assert len(v) > 0

    def test_gen_random_datetime_with_rng(self) -> None:
        v = _gen_random_datetime({}, self.rng)
        assert isinstance(v, datetime)

    def test_gen_random_date_with_rng(self) -> None:
        v = _gen_random_date({}, self.rng)
        assert isinstance(v, date)

    def test_gen_random_time_with_rng(self) -> None:
        v = _gen_random_time({}, self.rng)
        assert isinstance(v, time)

    def test_gen_random_choice_with_rng(self) -> None:
        v = _gen_random_choice({"enum_values": ["x", "y", "z"]}, self.rng)
        assert v in {"x", "y", "z"}

    def test_gen_random_bytes_with_rng(self) -> None:
        v = _gen_random_bytes({}, self.rng)
        assert isinstance(v, bytes)
        assert len(v) == 16

    def test_gen_random_json_with_rng(self) -> None:
        v = _gen_random_json({}, self.rng)
        assert isinstance(v, dict)

    def test_gen_random_list_with_rng(self) -> None:
        v = _gen_random_list({}, self.rng)
        assert isinstance(v, list)

    def test_gen_ssn_with_rng(self) -> None:
        v = _gen_ssn(self.rng)
        assert re.fullmatch(r"\d{3}-\d{2}-\d{4}", v)

    def test_rng_determinism(self) -> None:
        """Same rng seed → same output for each builtin."""
        rng_a = random.Random(7)  # noqa: S311
        rng_b = random.Random(7)  # noqa: S311
        assert _gen_random_int({"min": 0, "max": 1000}, rng_a) == _gen_random_int(
            {"min": 0, "max": 1000}, rng_b
        )
        assert _gen_random_string({"max_length": 20}, rng_a) == _gen_random_string(
            {"max_length": 20}, rng_b
        )

    def test_rng_isolation_between_columns(self) -> None:
        """Two different rng instances don't interfere with each other."""
        rng_col_a = random.Random(111)  # noqa: S311
        rng_col_b = random.Random(222)  # noqa: S311
        vals_a = [_gen_random_int({}, rng_col_a) for _ in range(5)]
        vals_b = [_gen_random_int({}, rng_col_b) for _ in range(5)]
        assert vals_a != vals_b
