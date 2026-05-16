"""Tests for dbsprout.schema.models — Pydantic schema models."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from dbsprout.schema.models import (
    _CONTROL_CHAR_RE,
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    IndexSchema,
    TableSchema,
)

# ── ColumnType Enum ──────────────────────────────────────────────────────


EXPECTED_TYPES = [
    ("INTEGER", "integer"),
    ("BIGINT", "bigint"),
    ("SMALLINT", "smallint"),
    ("FLOAT", "float"),
    ("DECIMAL", "decimal"),
    ("BOOLEAN", "boolean"),
    ("VARCHAR", "varchar"),
    ("TEXT", "text"),
    ("DATE", "date"),
    ("DATETIME", "datetime"),
    ("TIMESTAMP", "timestamp"),
    ("TIME", "time"),
    ("UUID", "uuid"),
    ("JSON", "json"),
    ("BINARY", "binary"),
    ("ENUM", "enum"),
    ("ARRAY", "array"),
    ("UNKNOWN", "unknown"),
]


def test_column_type_has_all_18_values() -> None:
    assert len(ColumnType) == 18


@pytest.mark.parametrize(("attr", "value"), EXPECTED_TYPES)
def test_column_type_values(attr: str, value: str) -> None:
    assert getattr(ColumnType, attr).value == value


def test_column_type_string_construction() -> None:
    assert ColumnType("integer") is ColumnType.INTEGER


def test_column_type_invalid_raises() -> None:
    with pytest.raises(ValueError, match="not_a_type"):
        ColumnType("not_a_type")


# ── ColumnSchema ─────────────────────────────────────────────────────────


def test_column_schema_minimal() -> None:
    col = ColumnSchema(name="id", data_type=ColumnType.INTEGER)
    assert col.name == "id"
    assert col.data_type is ColumnType.INTEGER
    assert col.nullable is True
    assert col.primary_key is False
    assert col.unique is False
    assert col.autoincrement is False
    assert col.default is None
    assert col.raw_type == ""
    assert col.max_length is None
    assert col.precision is None
    assert col.scale is None
    assert col.enum_values is None
    assert col.check_constraint is None
    assert col.comment is None


def test_column_schema_full() -> None:
    col = ColumnSchema(
        name="price",
        data_type=ColumnType.DECIMAL,
        raw_type="decimal(10,2)",
        nullable=False,
        primary_key=False,
        unique=False,
        autoincrement=False,
        default="0.00",
        max_length=None,
        precision=10,
        scale=2,
        enum_values=None,
        check_constraint="price >= 0",
        comment="Product price in USD",
    )
    assert col.precision == 10
    assert col.scale == 2
    assert col.check_constraint == "price >= 0"


def test_column_schema_frozen() -> None:
    col = ColumnSchema(name="id", data_type=ColumnType.INTEGER)
    with pytest.raises(ValidationError):
        col.name = "changed"  # type: ignore[misc]


def test_column_schema_json_round_trip() -> None:
    col = ColumnSchema(
        name="status",
        data_type=ColumnType.ENUM,
        enum_values=["active", "inactive"],
        nullable=False,
    )
    json_str = col.model_dump_json()
    restored = ColumnSchema.model_validate_json(json_str)
    assert restored == col
    assert restored.enum_values == ["active", "inactive"]


def test_column_schema_enum_values_preserved() -> None:
    col = ColumnSchema(
        name="status",
        data_type=ColumnType.ENUM,
        enum_values=["z_last", "a_first", "m_middle"],
    )
    assert col.enum_values == ["z_last", "a_first", "m_middle"]


# ── ForeignKeySchema ─────────────────────────────────────────────────────


def test_fk_schema_minimal() -> None:
    fk = ForeignKeySchema(
        columns=["user_id"],
        ref_table="users",
        ref_columns=["id"],
    )
    assert fk.name is None
    assert fk.columns == ["user_id"]
    assert fk.ref_table == "users"
    assert fk.ref_columns == ["id"]
    assert fk.on_delete is None
    assert fk.on_update is None
    assert fk.deferrable is False


def test_fk_schema_full() -> None:
    fk = ForeignKeySchema(
        name="fk_orders_user",
        columns=["user_id"],
        ref_table="users",
        ref_columns=["id"],
        on_delete="CASCADE",
        on_update="NO ACTION",
        deferrable=True,
    )
    assert fk.on_delete == "CASCADE"
    assert fk.deferrable is True


def test_fk_schema_json_round_trip() -> None:
    fk = ForeignKeySchema(
        name="fk_test",
        columns=["a", "b"],
        ref_table="parent",
        ref_columns=["x", "y"],
        on_delete="SET NULL",
    )
    restored = ForeignKeySchema.model_validate_json(fk.model_dump_json())
    assert restored == fk


def test_fk_schema_frozen() -> None:
    fk = ForeignKeySchema(columns=["a"], ref_table="t", ref_columns=["b"])
    with pytest.raises(ValidationError):
        fk.ref_table = "changed"  # type: ignore[misc]


# ── IndexSchema ──────────────────────────────────────────────────────────


def test_index_schema_basic() -> None:
    idx = IndexSchema(columns=["email"])
    assert idx.name is None
    assert idx.columns == ["email"]
    assert idx.unique is False


def test_index_schema_unique() -> None:
    idx = IndexSchema(name="idx_email", columns=["email"], unique=True)
    assert idx.unique is True


def test_index_schema_json_round_trip() -> None:
    idx = IndexSchema(name="idx_comp", columns=["a", "b"], unique=True)
    restored = IndexSchema.model_validate_json(idx.model_dump_json())
    assert restored == idx


# ── TableSchema ──────────────────────────────────────────────────────────


def _make_col(name: str, dtype: ColumnType = ColumnType.INTEGER, **kw: Any) -> ColumnSchema:
    return ColumnSchema(name=name, data_type=dtype, **kw)


def _make_fk(cols: list[str], ref: str, ref_cols: list[str], **kw: Any) -> ForeignKeySchema:
    return ForeignKeySchema(columns=cols, ref_table=ref, ref_columns=ref_cols, **kw)


def test_table_schema_basic() -> None:
    t = TableSchema(
        name="users",
        columns=[_make_col("id"), _make_col("name", ColumnType.VARCHAR)],
        primary_key=["id"],
    )
    assert t.name == "users"
    assert len(t.columns) == 2
    assert t.primary_key == ["id"]
    assert t.foreign_keys == []
    assert t.indexes == []
    assert t.comment is None
    assert t.row_count_hint is None


def test_table_get_column_found() -> None:
    t = TableSchema(
        name="t",
        columns=[_make_col("id"), _make_col("email", ColumnType.VARCHAR)],
    )
    col = t.get_column("email")
    assert col is not None
    assert col.name == "email"


def test_table_get_column_not_found() -> None:
    t = TableSchema(name="t", columns=[_make_col("id")])
    assert t.get_column("missing") is None


def test_fk_parent_tables_basic() -> None:
    t = TableSchema(
        name="orders",
        columns=[_make_col("id"), _make_col("user_id")],
        foreign_keys=[_make_fk(["user_id"], "users", ["id"])],
    )
    assert t.fk_parent_tables == ["users"]


def test_fk_parent_tables_self_ref_excluded() -> None:
    t = TableSchema(
        name="employees",
        columns=[_make_col("id"), _make_col("manager_id")],
        foreign_keys=[_make_fk(["manager_id"], "employees", ["id"])],
    )
    assert t.fk_parent_tables == []


def test_fk_parent_tables_duplicates_deduplicated() -> None:
    t = TableSchema(
        name="orders",
        columns=[_make_col("id"), _make_col("billing_id"), _make_col("shipping_id")],
        foreign_keys=[
            _make_fk(["billing_id"], "addresses", ["id"]),
            _make_fk(["shipping_id"], "addresses", ["id"]),
        ],
    )
    assert t.fk_parent_tables == ["addresses"]


def test_fk_parent_tables_sorted() -> None:
    t = TableSchema(
        name="t",
        columns=[_make_col("a"), _make_col("b")],
        foreign_keys=[
            _make_fk(["b"], "zebra", ["id"]),
            _make_fk(["a"], "alpha", ["id"]),
        ],
    )
    assert t.fk_parent_tables == ["alpha", "zebra"]


def test_fk_parent_tables_empty() -> None:
    t = TableSchema(name="t", columns=[_make_col("id")])
    assert t.fk_parent_tables == []


def test_is_junction_table_true() -> None:
    t = TableSchema(
        name="user_roles",
        columns=[_make_col("user_id"), _make_col("role_id")],
        primary_key=["user_id", "role_id"],
        foreign_keys=[
            _make_fk(["user_id"], "users", ["id"]),
            _make_fk(["role_id"], "roles", ["id"]),
        ],
    )
    assert t.is_junction_table is True


def test_is_junction_table_false_no_fks() -> None:
    t = TableSchema(
        name="users",
        columns=[_make_col("id")],
        primary_key=["id"],
    )
    assert t.is_junction_table is False


def test_is_junction_table_false_one_fk() -> None:
    t = TableSchema(
        name="orders",
        columns=[_make_col("id"), _make_col("user_id")],
        primary_key=["id"],
        foreign_keys=[_make_fk(["user_id"], "users", ["id"])],
    )
    assert t.is_junction_table is False


def test_is_junction_table_false_surrogate_pk() -> None:
    t = TableSchema(
        name="user_roles",
        columns=[_make_col("id"), _make_col("user_id"), _make_col("role_id")],
        primary_key=["id"],
        foreign_keys=[
            _make_fk(["user_id"], "users", ["id"]),
            _make_fk(["role_id"], "roles", ["id"]),
        ],
    )
    assert t.is_junction_table is False


def test_is_junction_table_false_empty_pk() -> None:
    t = TableSchema(
        name="t",
        columns=[_make_col("a"), _make_col("b")],
        primary_key=[],
        foreign_keys=[
            _make_fk(["a"], "x", ["id"]),
            _make_fk(["b"], "y", ["id"]),
        ],
    )
    assert t.is_junction_table is False


# ── DatabaseSchema ───────────────────────────────────────────────────────


def _users_table() -> TableSchema:
    return TableSchema(
        name="users",
        columns=[
            _make_col("id", primary_key=True),
            _make_col("email", ColumnType.VARCHAR, nullable=False, unique=True),
        ],
        primary_key=["id"],
    )


def _posts_table() -> TableSchema:
    return TableSchema(
        name="posts",
        columns=[
            _make_col("id", primary_key=True),
            _make_col("user_id", nullable=False),
            _make_col("title", ColumnType.VARCHAR, nullable=False),
        ],
        primary_key=["id"],
        foreign_keys=[_make_fk(["user_id"], "users", ["id"], on_delete="CASCADE")],
    )


def _simple_db() -> DatabaseSchema:
    return DatabaseSchema(tables=[_users_table(), _posts_table()])


def test_database_schema_basic() -> None:
    db = _simple_db()
    assert len(db.tables) == 2
    assert db.enums == {}
    assert db.dialect is None
    assert db.source is None
    assert db.source_file is None


def test_get_table_found() -> None:
    db = _simple_db()
    t = db.get_table("users")
    assert t is not None
    assert t.name == "users"


def test_get_table_not_found() -> None:
    db = _simple_db()
    assert db.get_table("missing") is None


def test_table_names() -> None:
    db = _simple_db()
    assert db.table_names() == ["users", "posts"]


def test_table_names_empty() -> None:
    db = DatabaseSchema(tables=[])
    assert db.table_names() == []


def test_schema_hash_deterministic() -> None:
    db1 = _simple_db()
    db2 = _simple_db()
    assert db1.schema_hash() == db2.schema_hash()


def test_schema_hash_different_schemas_differ() -> None:
    db1 = _simple_db()
    db2 = DatabaseSchema(tables=[_users_table()])
    assert db1.schema_hash() != db2.schema_hash()


def test_schema_hash_table_order_independent() -> None:
    db_ab = DatabaseSchema(tables=[_users_table(), _posts_table()])
    db_ba = DatabaseSchema(tables=[_posts_table(), _users_table()])
    assert db_ab.schema_hash() == db_ba.schema_hash()


def test_schema_hash_column_order_independent() -> None:
    cols_ab = [_make_col("a"), _make_col("b")]
    cols_ba = [_make_col("b"), _make_col("a")]
    db1 = DatabaseSchema(tables=[TableSchema(name="t", columns=cols_ab)])
    db2 = DatabaseSchema(tables=[TableSchema(name="t", columns=cols_ba)])
    assert db1.schema_hash() == db2.schema_hash()


def test_schema_hash_is_16_hex() -> None:
    h = _simple_db().schema_hash()
    assert len(h) == 16
    int(h, 16)  # raises ValueError if not valid hex


def test_schema_hash_ignores_metadata() -> None:
    db1 = DatabaseSchema(tables=[_users_table()], dialect="postgresql", source="introspect")
    db2 = DatabaseSchema(tables=[_users_table()], dialect="sqlite", source="file")
    assert db1.schema_hash() == db2.schema_hash()


def test_to_ddl_generates_create_table() -> None:
    ddl = _simple_db().to_ddl()
    assert "CREATE TABLE" in ddl
    assert "users" in ddl
    assert "posts" in ddl


def test_to_ddl_includes_fk_constraints() -> None:
    ddl = _simple_db().to_ddl()
    assert "FOREIGN KEY" in ddl
    assert "REFERENCES" in ddl
    assert "users" in ddl


def test_to_ddl_includes_not_null_and_unique() -> None:
    ddl = _simple_db().to_ddl()
    assert "NOT NULL" in ddl
    assert "UNIQUE" in ddl


def test_database_schema_json_round_trip() -> None:
    db = DatabaseSchema(
        tables=[_users_table(), _posts_table()],
        enums={"status": ["active", "inactive"]},
        dialect="postgresql",
        source="test",
    )
    restored = DatabaseSchema.model_validate_json(db.model_dump_json())
    assert restored == db


def test_database_schema_json_round_trip_none_fields() -> None:
    db = DatabaseSchema(tables=[])
    restored = DatabaseSchema.model_validate_json(db.model_dump_json())
    assert restored == db
    assert restored.dialect is None


def test_database_schema_frozen() -> None:
    db = _simple_db()
    with pytest.raises(ValidationError):
        db.dialect = "changed"  # type: ignore[misc]


def test_model_copy_produces_new_instance() -> None:
    db = _simple_db()
    db2 = db.model_copy(update={"dialect": "postgresql"})
    assert db.dialect is None
    assert db2.dialect == "postgresql"
    assert db is not db2


# ── DDL edge cases ───────────────────────────────────────────────────────


def test_to_ddl_varchar_with_max_length() -> None:
    db = DatabaseSchema(
        tables=[
            TableSchema(
                name="t",
                columns=[_make_col("email", ColumnType.VARCHAR, max_length=255)],
            )
        ]
    )
    assert "VARCHAR(255)" in db.to_ddl()


def test_to_ddl_decimal_with_precision_and_scale() -> None:
    db = DatabaseSchema(
        tables=[
            TableSchema(
                name="t",
                columns=[_make_col("price", ColumnType.DECIMAL, precision=10, scale=2)],
            )
        ]
    )
    assert "DECIMAL(10,2)" in db.to_ddl()


def test_to_ddl_decimal_precision_only() -> None:
    db = DatabaseSchema(
        tables=[
            TableSchema(
                name="t",
                columns=[_make_col("amount", ColumnType.DECIMAL, precision=8)],
            )
        ]
    )
    assert "DECIMAL(8)" in db.to_ddl()


def test_to_ddl_default_and_check() -> None:
    db = DatabaseSchema(
        tables=[
            TableSchema(
                name="t",
                columns=[
                    _make_col(
                        "age",
                        default="0",
                        check_constraint="age >= 0",
                        nullable=False,
                    ),
                ],
            )
        ]
    )
    ddl = db.to_ddl()
    assert "DEFAULT 0" in ddl
    assert "CHECK (age >= 0)" in ddl


def test_to_ddl_composite_pk() -> None:
    db = DatabaseSchema(
        tables=[
            TableSchema(
                name="user_roles",
                columns=[_make_col("user_id"), _make_col("role_id")],
                primary_key=["user_id", "role_id"],
            )
        ]
    )
    ddl = db.to_ddl()
    assert 'PRIMARY KEY ("user_id", "role_id")' in ddl


def test_to_ddl_fk_on_update() -> None:
    db = DatabaseSchema(
        tables=[
            TableSchema(
                name="t",
                columns=[_make_col("ref_id")],
                foreign_keys=[
                    _make_fk(["ref_id"], "parent", ["id"], on_update="CASCADE"),
                ],
            )
        ]
    )
    assert "ON UPDATE CASCADE" in db.to_ddl()


# ── schema_hash canonical coverage ───────────────────────────────────────


def test_schema_hash_includes_optional_column_fields() -> None:
    """Columns with default, max_length, precision, scale, enum_values, check differ."""
    col_plain = _make_col("x")
    col_rich = _make_col(
        "x",
        default="42",
        max_length=100,
        precision=5,
        scale=2,
        check_constraint="x > 0",
    )
    db1 = DatabaseSchema(tables=[TableSchema(name="t", columns=[col_plain])])
    db2 = DatabaseSchema(tables=[TableSchema(name="t", columns=[col_rich])])
    assert db1.schema_hash() != db2.schema_hash()


def test_schema_hash_enum_values_included() -> None:
    col1 = ColumnSchema(name="s", data_type=ColumnType.ENUM, enum_values=["a", "b"])
    col2 = ColumnSchema(name="s", data_type=ColumnType.ENUM, enum_values=["c", "d"])
    db1 = DatabaseSchema(tables=[TableSchema(name="t", columns=[col1])])
    db2 = DatabaseSchema(tables=[TableSchema(name="t", columns=[col2])])
    assert db1.schema_hash() != db2.schema_hash()


def test_schema_hash_ignores_raw_type() -> None:
    col1 = _make_col("x", raw_type="int4")
    col2 = _make_col("x", raw_type="integer")
    db1 = DatabaseSchema(tables=[TableSchema(name="t", columns=[col1])])
    db2 = DatabaseSchema(tables=[TableSchema(name="t", columns=[col2])])
    assert db1.schema_hash() == db2.schema_hash()


def test_schema_hash_ignores_comments() -> None:
    col1 = _make_col("x", comment="user id")
    col2 = _make_col("x", comment="identifier")
    t1 = TableSchema(name="t", columns=[col1], comment="table A")
    t2 = TableSchema(name="t", columns=[col2], comment="table B")
    db1 = DatabaseSchema(tables=[t1])
    db2 = DatabaseSchema(tables=[t2])
    assert db1.schema_hash() == db2.schema_hash()


def test_schema_hash_ignores_row_count_hint() -> None:
    t1 = TableSchema(name="t", columns=[_make_col("x")], row_count_hint=100)
    t2 = TableSchema(name="t", columns=[_make_col("x")], row_count_hint=999)
    db1 = DatabaseSchema(tables=[t1])
    db2 = DatabaseSchema(tables=[t2])
    assert db1.schema_hash() == db2.schema_hash()


def test_schema_hash_ignores_fk_and_index_names() -> None:
    fk1 = _make_fk(["a"], "p", ["id"], name="fk_old")
    fk2 = _make_fk(["a"], "p", ["id"], name="fk_new")
    idx1 = IndexSchema(name="idx_old", columns=["x"])
    idx2 = IndexSchema(name="idx_new", columns=["x"])
    t1 = TableSchema(name="t", columns=[_make_col("a")], foreign_keys=[fk1], indexes=[idx1])
    t2 = TableSchema(name="t", columns=[_make_col("a")], foreign_keys=[fk2], indexes=[idx2])
    db1 = DatabaseSchema(tables=[t1])
    db2 = DatabaseSchema(tables=[t2])
    assert db1.schema_hash() == db2.schema_hash()


def test_index_schema_frozen() -> None:
    idx = IndexSchema(columns=["a"])
    with pytest.raises(ValidationError):
        idx.unique = True  # type: ignore[misc]


def test_table_schema_frozen() -> None:
    t = TableSchema(name="t", columns=[_make_col("id")])
    with pytest.raises(ValidationError):
        t.name = "changed"  # type: ignore[misc]


def test_to_ddl_autoincrement() -> None:
    db = DatabaseSchema(
        tables=[
            TableSchema(
                name="t",
                columns=[_make_col("id", autoincrement=True, primary_key=True)],
                primary_key=["id"],
            )
        ]
    )
    assert "AUTOINCREMENT" in db.to_ddl()


def test_to_ddl_with_dialect_param() -> None:
    db = DatabaseSchema(tables=[TableSchema(name="t", columns=[_make_col("id")])])
    ddl = db.to_ddl(dialect="postgresql")
    assert "CREATE TABLE" in ddl


# ── Identifier validation ───────────────────────────────────────────────


class TestIdentifierValidation:
    """AC: ColumnSchema.name and TableSchema.name validated."""

    def test_column_name_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ColumnSchema(name="", data_type=ColumnType.INTEGER)

    def test_column_name_whitespace_only_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ColumnSchema(name="   ", data_type=ColumnType.INTEGER)

    def test_column_name_control_char_null_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ColumnSchema(name="id\x00", data_type=ColumnType.INTEGER)

    def test_column_name_control_char_newline_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ColumnSchema(name="id\n", data_type=ColumnType.INTEGER)

    def test_column_name_control_char_tab_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ColumnSchema(name="id\t", data_type=ColumnType.INTEGER)

    def test_column_name_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ColumnSchema(name="x" * 129, data_type=ColumnType.INTEGER)

    def test_column_name_128_chars_accepted(self) -> None:
        col = ColumnSchema(name="x" * 128, data_type=ColumnType.INTEGER)
        assert len(col.name) == 128

    @pytest.mark.parametrize(
        "name",
        ["id", "user_name", "名前", "my column", "my.column", "order", "col-1"],
    )
    def test_column_name_valid_cases(self, name: str) -> None:
        col = ColumnSchema(name=name, data_type=ColumnType.INTEGER)
        assert col.name == name

    def test_column_name_whitespace_stripped(self) -> None:
        col = ColumnSchema(name="  id  ", data_type=ColumnType.INTEGER)
        assert col.name == "id"

    def test_table_name_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TableSchema(name="", columns=[_make_col("id")])

    def test_table_name_control_chars_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TableSchema(name="t\x00", columns=[_make_col("id")])

    @pytest.mark.parametrize("bad_name", ["../foo", "a/b", "a\\b", "..", "."])
    def test_path_traversal_chars_rejected_on_table(self, bad_name: str) -> None:
        with pytest.raises(ValidationError):
            TableSchema(name=bad_name, columns=[_make_col("id")])

    @pytest.mark.parametrize("bad_name", ["../foo", "a/b", "a\\b", "..", "."])
    def test_path_traversal_chars_rejected_on_column(self, bad_name: str) -> None:
        with pytest.raises(ValidationError):
            ColumnSchema(name=bad_name, data_type=ColumnType.INTEGER)


# ── Referential action validation ───────────────────────────────────────


class TestReferentialAction:
    """AC: on_delete/on_update constrained to valid SQL referential actions."""

    @pytest.mark.parametrize(
        "action",
        ["CASCADE", "SET NULL", "SET DEFAULT", "RESTRICT", "NO ACTION"],
    )
    def test_on_delete_valid_values(self, action: str) -> None:
        fk = ForeignKeySchema(columns=["a"], ref_table="t", ref_columns=["b"], on_delete=action)
        assert fk.on_delete == action

    def test_on_delete_none_accepted(self) -> None:
        fk = ForeignKeySchema(columns=["a"], ref_table="t", ref_columns=["b"])
        assert fk.on_delete is None

    def test_on_delete_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ForeignKeySchema(columns=["a"], ref_table="t", ref_columns=["b"], on_delete="MAGIC")

    def test_on_delete_case_insensitive(self) -> None:
        fk = ForeignKeySchema(columns=["a"], ref_table="t", ref_columns=["b"], on_delete="cascade")
        assert fk.on_delete == "CASCADE"

    @pytest.mark.parametrize(
        "action",
        ["CASCADE", "SET NULL", "SET DEFAULT", "RESTRICT", "NO ACTION"],
    )
    def test_on_update_valid_values(self, action: str) -> None:
        fk = ForeignKeySchema(columns=["a"], ref_table="t", ref_columns=["b"], on_update=action)
        assert fk.on_update == action

    def test_on_update_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ForeignKeySchema(columns=["a"], ref_table="t", ref_columns=["b"], on_update="INVALID")


# ── DDL identifier quoting ──────────────────────────────────────────────


class TestDDLQuoting:
    """AC: to_ddl() double-quotes all identifiers."""

    def test_table_name_quoted(self) -> None:
        db = DatabaseSchema(tables=[TableSchema(name="users", columns=[_make_col("id")])])
        assert 'CREATE TABLE "users"' in db.to_ddl()

    def test_column_name_quoted(self) -> None:
        db = DatabaseSchema(
            tables=[
                TableSchema(
                    name="t",
                    columns=[
                        _make_col("id"),
                        _make_col("email", ColumnType.VARCHAR),
                    ],
                )
            ]
        )
        ddl = db.to_ddl()
        assert '"id"' in ddl
        assert '"email"' in ddl

    def test_composite_pk_columns_quoted(self) -> None:
        db = DatabaseSchema(
            tables=[
                TableSchema(
                    name="t",
                    columns=[_make_col("a"), _make_col("b")],
                    primary_key=["a", "b"],
                )
            ]
        )
        assert 'PRIMARY KEY ("a", "b")' in db.to_ddl()

    def test_fk_columns_quoted(self) -> None:
        db = DatabaseSchema(
            tables=[
                TableSchema(
                    name="t",
                    columns=[_make_col("user_id")],
                    foreign_keys=[_make_fk(["user_id"], "users", ["id"])],
                )
            ]
        )
        ddl = db.to_ddl()
        assert 'FOREIGN KEY ("user_id")' in ddl
        assert 'REFERENCES "users" ("id")' in ddl

    def test_internal_double_quote_escaped(self) -> None:
        db = DatabaseSchema(
            tables=[
                TableSchema(
                    name='table"name',
                    columns=[_make_col('col"umn')],
                )
            ]
        )
        ddl = db.to_ddl()
        assert '"table""name"' in ddl
        assert '"col""umn"' in ddl

    def test_reserved_word_name_quoted(self) -> None:
        db = DatabaseSchema(tables=[TableSchema(name="order", columns=[_make_col("select")])])
        ddl = db.to_ddl()
        assert 'CREATE TABLE "order"' in ddl
        assert '"select"' in ddl


# ── Adversarial input tests ─────────────────────────────────────────────


class TestAdversarialInput:
    """AC: adversarial input test cases for name fields, check_constraint, default."""

    def test_control_char_null_byte_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ColumnSchema(name="id\x00", data_type=ColumnType.INTEGER)

    def test_control_char_newline_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ColumnSchema(name="id\n", data_type=ColumnType.INTEGER)

    def test_control_char_tab_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ColumnSchema(name="id\t", data_type=ColumnType.INTEGER)

    def test_control_char_del_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ColumnSchema(name="id\x7f", data_type=ColumnType.INTEGER)

    def test_adversarial_name_accepted_and_quoted_in_ddl(self) -> None:
        """SQL injection payload is allowed as a name but safely quoted in DDL."""
        name = "'; DROP TABLE x; --"
        col = ColumnSchema(name=name, data_type=ColumnType.INTEGER)
        assert col.name == name
        db = DatabaseSchema(tables=[TableSchema(name="safe", columns=[col])])
        ddl = db.to_ddl()
        assert '"\'; DROP TABLE x; --"' in ddl

    def test_name_with_embedded_quotes_escaped_in_ddl(self) -> None:
        db = DatabaseSchema(
            tables=[
                TableSchema(
                    name='tab"le',
                    columns=[_make_col('col"umn')],
                )
            ]
        )
        ddl = db.to_ddl()
        assert '"tab""le"' in ddl
        assert '"col""umn"' in ddl

    def test_check_constraint_passthrough(self) -> None:
        """check_constraint is trusted input — passes through without validation."""
        col = ColumnSchema(
            name="x",
            data_type=ColumnType.INTEGER,
            check_constraint="1=1) OR (1=1",
        )
        assert col.check_constraint == "1=1) OR (1=1"

    def test_default_passthrough(self) -> None:
        """default is trusted input — passes through without validation."""
        col = ColumnSchema(
            name="x",
            data_type=ColumnType.INTEGER,
            default="'; DROP TABLE x; --",
        )
        assert col.default == "'; DROP TABLE x; --"

    def test_empty_name_after_strip_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ColumnSchema(name="   ", data_type=ColumnType.INTEGER)

    def test_max_length_boundary_129_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ColumnSchema(name="x" * 129, data_type=ColumnType.INTEGER)

    def test_max_length_boundary_128_accepted(self) -> None:
        col = ColumnSchema(name="y" * 128, data_type=ColumnType.INTEGER)
        assert len(col.name) == 128

    def test_invalid_referential_action_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ForeignKeySchema(
                columns=["a"],
                ref_table="t",
                ref_columns=["b"],
                on_delete="DROP TABLE users",
            )


# ── C1 control-char range ───────────────────────────────────────────────


class TestC1ControlChars:
    """AC: control-char regex blocks C1 range U+0080-U+009F."""

    @pytest.mark.parametrize("cp", [0x80, 0x85, 0x90, 0x9F])
    def test_c1_control_char_rejected_on_column(self, cp: int) -> None:
        with pytest.raises(ValidationError):
            ColumnSchema(name=f"id{chr(cp)}", data_type=ColumnType.INTEGER)

    def test_c1_lower_boundary_0x80_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TableSchema(name=f"t{chr(0x80)}", columns=[_make_col("id")])

    def test_c1_upper_boundary_0x9f_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TableSchema(name=f"t{chr(0x9F)}", columns=[_make_col("id")])

    def test_char_past_c1_range_still_allowed(self) -> None:
        """U+00E0 (à, just past the C1 control range) remains a valid identifier."""
        col = ColumnSchema(name=f"id{chr(0xE0)}", data_type=ColumnType.INTEGER)
        assert chr(0xE0) in col.name

    def test_regex_boundary_excludes_0xa0(self) -> None:
        """The control-char regex must not match U+00A0 (just past C1)."""
        assert _CONTROL_CHAR_RE.search(chr(0xA0)) is None
        assert _CONTROL_CHAR_RE.search(chr(0x9F)) is not None
        assert _CONTROL_CHAR_RE.search(chr(0x80)) is not None


# ── DeferTiming validation ──────────────────────────────────────────────


class TestDeferTiming:
    """AC: ForeignKeySchema.initially constrained to DEFERRED|IMMEDIATE|None."""

    @pytest.mark.parametrize("val", ["DEFERRED", "IMMEDIATE"])
    def test_valid_values(self, val: str) -> None:
        fk = ForeignKeySchema(columns=["a"], ref_table="t", ref_columns=["b"], initially=val)
        assert fk.initially == val

    def test_none_accepted(self) -> None:
        fk = ForeignKeySchema(columns=["a"], ref_table="t", ref_columns=["b"])
        assert fk.initially is None

    def test_case_insensitive_normalized(self) -> None:
        fk = ForeignKeySchema(columns=["a"], ref_table="t", ref_columns=["b"], initially="deferred")
        assert fk.initially == "DEFERRED"

    def test_invalid_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ForeignKeySchema(columns=["a"], ref_table="t", ref_columns=["b"], initially="SOON")

    def test_injection_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ForeignKeySchema(
                columns=["a"],
                ref_table="t",
                ref_columns=["b"],
                initially="DROP TABLE x",
            )
