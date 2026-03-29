"""Tests for dbsprout.spec.heuristics — column-to-generator mapping."""

from __future__ import annotations

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)
from dbsprout.spec.heuristics import map_columns
from dbsprout.spec.patterns import PATTERNS


def _col(name: str, dtype: ColumnType = ColumnType.VARCHAR, **kw: object) -> ColumnSchema:
    return ColumnSchema(name=name, data_type=dtype, **kw)  # type: ignore[arg-type]


def _table(name: str, columns: list[ColumnSchema]) -> TableSchema:
    return TableSchema(name=name, columns=columns)


def _schema(*tables: TableSchema) -> DatabaseSchema:
    return DatabaseSchema(tables=list(tables))


# ── Pattern matching ─────────────────────────────────────────────────────


class TestEmailMapping:
    def test_email_column(self) -> None:
        schema = _schema(_table("users", [_col("id", ColumnType.INTEGER), _col("email")]))
        result = map_columns(schema)
        m = result["users"]["email"]
        assert m.generator_name == "email"
        assert m.confidence >= 0.9

    def test_email_address_column(self) -> None:
        schema = _schema(_table("t", [_col("email_address")]))
        result = map_columns(schema)
        assert result["t"]["email_address"].generator_name == "email"


class TestNameMapping:
    def test_first_name(self) -> None:
        schema = _schema(_table("t", [_col("first_name")]))
        result = map_columns(schema)
        assert result["t"]["first_name"].generator_name == "first_name"
        assert result["t"]["first_name"].confidence >= 0.9

    def test_camel_case(self) -> None:
        schema = _schema(_table("t", [_col("firstName")]))
        result = map_columns(schema)
        assert result["t"]["firstName"].generator_name == "first_name"

    def test_email_address_camel(self) -> None:
        """emailAddress → email_address via token normalization (step 2)."""
        schema = _schema(_table("t", [_col("emailAddress")]))
        result = map_columns(schema)
        assert result["t"]["emailAddress"].generator_name == "email"
        assert result["t"]["emailAddress"].confidence < 0.95  # reduced by 0.9 factor

    def test_last_name(self) -> None:
        schema = _schema(_table("t", [_col("last_name")]))
        result = map_columns(schema)
        assert result["t"]["last_name"].generator_name == "last_name"


class TestTemporalMapping:
    def test_created_at(self) -> None:
        schema = _schema(_table("t", [_col("created_at", ColumnType.TIMESTAMP)]))
        result = map_columns(schema)
        assert result["t"]["created_at"].generator_name == "datetime"
        assert result["t"]["created_at"].confidence >= 0.85

    def test_updated_at(self) -> None:
        schema = _schema(_table("t", [_col("updated_at", ColumnType.TIMESTAMP)]))
        result = map_columns(schema)
        assert result["t"]["updated_at"].generator_name == "datetime"


class TestFinancialMapping:
    def test_price(self) -> None:
        schema = _schema(_table("t", [_col("price", ColumnType.DECIMAL)]))
        result = map_columns(schema)
        assert result["t"]["price"].generator_name == "price"
        assert result["t"]["price"].confidence >= 0.85


class TestAddressMapping:
    def test_city(self) -> None:
        schema = _schema(_table("t", [_col("city")]))
        result = map_columns(schema)
        assert result["t"]["city"].generator_name == "city"

    def test_country(self) -> None:
        schema = _schema(_table("t", [_col("country")]))
        result = map_columns(schema)
        assert result["t"]["country"].generator_name == "country"


# ── Type fallback ────────────────────────────────────────────────────────


class TestTypeFallback:
    def test_unknown_integer(self) -> None:
        schema = _schema(_table("t", [_col("xyzzy", ColumnType.INTEGER)]))
        result = map_columns(schema)
        m = result["t"]["xyzzy"]
        assert m.generator_name == "random_int"
        assert m.confidence == 0.5

    def test_boolean_fallback(self) -> None:
        schema = _schema(_table("t", [_col("flag", ColumnType.BOOLEAN)]))
        result = map_columns(schema)
        assert result["t"]["flag"].generator_name == "random_bool"

    def test_uuid_fallback(self) -> None:
        schema = _schema(_table("t", [_col("qwxyz", ColumnType.UUID)]))
        result = map_columns(schema)
        assert result["t"]["qwxyz"].generator_name == "uuid4"

    def test_timestamp_fallback(self) -> None:
        schema = _schema(_table("t", [_col("ts", ColumnType.TIMESTAMP)]))
        result = map_columns(schema)
        assert result["t"]["ts"].generator_name == "random_datetime"
        assert result["t"]["ts"].confidence == 0.5

    def test_enum_with_values(self) -> None:
        schema = _schema(_table("t", [_col("qwxyz", ColumnType.ENUM, enum_values=["a", "b", "c"])]))
        result = map_columns(schema)
        m = result["t"]["qwxyz"]
        assert m.generator_name == "random_choice"
        assert m.params.get("enum_values") == ["a", "b", "c"]


class TestVarcharParams:
    def test_max_length_in_params(self) -> None:
        schema = _schema(_table("t", [_col("name", ColumnType.VARCHAR, max_length=100)]))
        result = map_columns(schema)
        assert result["t"]["name"].params.get("max_length") == 100


# ── Multi-table + pattern count ──────────────────────────────────────────


class TestMultiTable:
    def test_all_tables_mapped(self) -> None:
        schema = _schema(
            _table("users", [_col("id", ColumnType.INTEGER), _col("email"), _col("name")]),
            _table(
                "posts",
                [_col("id", ColumnType.INTEGER), _col("title"), _col("body", ColumnType.TEXT)],
            ),
        )
        result = map_columns(schema)
        assert "users" in result
        assert "posts" in result
        assert len(result["users"]) == 3
        assert len(result["posts"]) == 3


class TestDecimalParams:
    def test_precision_and_scale(self) -> None:
        schema = _schema(_table("t", [_col("amount", ColumnType.DECIMAL, precision=10, scale=2)]))
        result = map_columns(schema)
        assert result["t"]["amount"].params.get("precision") == 10
        assert result["t"]["amount"].params.get("scale") == 2


class TestPatternCount:
    def test_at_least_80_patterns(self) -> None:
        assert len(PATTERNS) >= 80
