"""Tests for dbsprout.schema.introspect — SQLite introspection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest
import sqlalchemy as sa

from dbsprout.schema import introspect as introspect_fn
from dbsprout.schema import normalize_type as normalize_type_fn
from dbsprout.schema.introspect import (
    _CONNECT_TIMEOUT,
    _collect_unique_columns,
    _create_engine,
    _detect_autoincrement,
    _extract_check_enum,
    _get_raw_type,
    _is_safe_identifier,
    _validate_url,
    introspect,
)
from dbsprout.schema.models import ColumnType, DatabaseSchema

# ── 7-table DDL fixture ─────────────────────────────────────────────────

SQLITE_DDL = """\
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    age INTEGER,
    is_active BOOLEAN NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE categories (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER,
    FOREIGN KEY (parent_id) REFERENCES categories (id) ON DELETE SET NULL
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    category_id INTEGER NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('draft', 'active', 'archived')),
    weight REAL,
    FOREIGN KEY (category_id) REFERENCES categories (id) ON DELETE CASCADE
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    total DECIMAL(12, 2) NOT NULL DEFAULT 0,
    notes TEXT,
    ordered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE INDEX idx_orders_user_id ON orders (user_id);

CREATE TABLE order_items (
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 1,
    unit_price DECIMAL(10, 2) NOT NULL,
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (order_id) REFERENCES orders (id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products (id) ON DELETE RESTRICT
);

CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    label VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE product_tags (
    product_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    PRIMARY KEY (product_id, tag_id),
    FOREIGN KEY (product_id) REFERENCES products (id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags (id) ON DELETE CASCADE
);
"""


def _execute_ddl(engine: sa.engine.Engine) -> None:
    """Execute the multi-statement DDL on the given engine."""
    with engine.begin() as conn:
        for statement in SQLITE_DDL.split(";"):
            cleaned = statement.strip()
            if cleaned:
                conn.execute(sa.text(cleaned))


@pytest.fixture
def sqlite_url() -> Generator[str, None, None]:
    """Create an in-memory SQLite database with the 7-table schema and return the URL."""
    url = "sqlite:///file:test_s003?mode=memory&cache=shared&uri=true"
    shared_engine = sa.create_engine(url)
    _execute_ddl(shared_engine)
    # Keep a connection open to prevent the shared-memory DB from being garbage-collected.
    keepalive_conn = shared_engine.connect()
    yield url
    keepalive_conn.close()
    shared_engine.dispose()


# ── Task 4: First test ──────────────────────────────────────────────────


class TestIntrospectReturnType:
    def test_returns_database_schema(self, sqlite_url: str) -> None:
        result = introspect(sqlite_url)
        assert isinstance(result, DatabaseSchema)

    def test_table_count(self, sqlite_url: str) -> None:
        result = introspect(sqlite_url)
        assert len(result.tables) == 7


# ── Task 5: Basic extraction tests ──────────────────────────────────────


class TestTableNames:
    def test_all_tables_present(self, sqlite_url: str) -> None:
        result = introspect(sqlite_url)
        names = sorted(result.table_names())
        assert names == [
            "categories",
            "order_items",
            "orders",
            "product_tags",
            "products",
            "tags",
            "users",
        ]


class TestDialectAndSource:
    def test_dialect_is_sqlite(self, sqlite_url: str) -> None:
        result = introspect(sqlite_url)
        assert result.dialect == "sqlite"

    def test_source_is_introspect(self, sqlite_url: str) -> None:
        result = introspect(sqlite_url)
        assert result.source == "introspect"


class TestColumnExtraction:
    def test_users_column_names(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        users = schema.get_table("users")
        assert users is not None
        col_names = [c.name for c in users.columns]
        assert col_names == ["id", "username", "email", "age", "is_active", "created_at"]

    def test_users_email_type(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        users = schema.get_table("users")
        assert users is not None
        email = users.get_column("email")
        assert email is not None
        assert email.data_type is ColumnType.VARCHAR
        assert email.max_length == 255

    def test_users_age_nullable(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        users = schema.get_table("users")
        assert users is not None
        age = users.get_column("age")
        assert age is not None
        assert age.nullable is True

    def test_users_username_not_nullable(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        users = schema.get_table("users")
        assert users is not None
        username = users.get_column("username")
        assert username is not None
        assert username.nullable is False

    def test_products_price_decimal(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        products = schema.get_table("products")
        assert products is not None
        price = products.get_column("price")
        assert price is not None
        assert price.data_type is ColumnType.DECIMAL
        assert price.precision == 10
        assert price.scale == 2


class TestPrimaryKeyExtraction:
    def test_users_single_pk(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        users = schema.get_table("users")
        assert users is not None
        assert users.primary_key == ["id"]

    def test_order_items_composite_pk(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        oi = schema.get_table("order_items")
        assert oi is not None
        assert sorted(oi.primary_key) == ["order_id", "product_id"]


class TestRawTypePreserved:
    def test_users_email_raw_type(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        users = schema.get_table("users")
        assert users is not None
        email = users.get_column("email")
        assert email is not None
        assert email.raw_type == "VARCHAR(255)"

    def test_products_weight_raw_type(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        products = schema.get_table("products")
        assert products is not None
        weight = products.get_column("weight")
        assert weight is not None
        assert weight.raw_type == "REAL"


# ── Task 7: FK + constraint tests ───────────────────────────────────────


class TestForeignKeyExtraction:
    def test_products_category_fk(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        products = schema.get_table("products")
        assert products is not None
        cat_fk = [fk for fk in products.foreign_keys if fk.ref_table == "categories"]
        assert len(cat_fk) == 1
        assert cat_fk[0].columns == ["category_id"]
        assert cat_fk[0].ref_columns == ["id"]

    def test_fk_on_delete_cascade(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        products = schema.get_table("products")
        assert products is not None
        cat_fk = [fk for fk in products.foreign_keys if fk.ref_table == "categories"]
        assert len(cat_fk) == 1
        assert cat_fk[0].on_delete == "CASCADE"

    def test_self_referencing_fk(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        categories = schema.get_table("categories")
        assert categories is not None
        self_fks = [fk for fk in categories.foreign_keys if fk.ref_table == "categories"]
        assert len(self_fks) == 1
        assert self_fks[0].columns == ["parent_id"]
        assert self_fks[0].on_delete == "SET NULL"

    def test_order_items_has_two_fks(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        oi = schema.get_table("order_items")
        assert oi is not None
        assert len(oi.foreign_keys) == 2

    def test_order_items_is_junction_table(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        oi = schema.get_table("order_items")
        assert oi is not None
        assert oi.is_junction_table is True

    def test_product_tags_is_junction_table(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        pt = schema.get_table("product_tags")
        assert pt is not None
        assert pt.is_junction_table is True


class TestUniqueConstraints:
    def test_users_email_unique(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        users = schema.get_table("users")
        assert users is not None
        email = users.get_column("email")
        assert email is not None
        assert email.unique is True

    def test_users_username_unique(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        users = schema.get_table("users")
        assert users is not None
        username = users.get_column("username")
        assert username is not None
        assert username.unique is True

    def test_tags_label_unique(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        tags = schema.get_table("tags")
        assert tags is not None
        label = tags.get_column("label")
        assert label is not None
        assert label.unique is True


class TestIndexExtraction:
    def test_orders_user_id_index(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        orders = schema.get_table("orders")
        assert orders is not None
        idx_names = [idx.name for idx in orders.indexes]
        assert "idx_orders_user_id" in idx_names
        idx = next(i for i in orders.indexes if i.name == "idx_orders_user_id")
        assert idx.columns == ["user_id"]
        assert idx.unique is False


class TestAutoincrementDetection:
    def test_users_id_autoincrement(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        users = schema.get_table("users")
        assert users is not None
        id_col = users.get_column("id")
        assert id_col is not None
        assert id_col.autoincrement is True

    def test_composite_pk_no_autoincrement(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        oi = schema.get_table("order_items")
        assert oi is not None
        for col in oi.columns:
            if col.primary_key:
                assert col.autoincrement is False, f"{col.name} should not be autoincrement"


# ── Task 9: Edge case tests ─────────────────────────────────────────────


class TestEmptyDatabase:
    def test_empty_db_returns_empty_schema(self) -> None:
        url = "sqlite:///file:test_empty?mode=memory&cache=shared&uri=true"
        engine = sa.create_engine(url)
        keepalive_conn = engine.connect()
        try:
            schema = introspect(url)
            assert isinstance(schema, DatabaseSchema)
            assert len(schema.tables) == 0
            assert schema.dialect == "sqlite"
            assert schema.source == "introspect"
        finally:
            keepalive_conn.close()
            engine.dispose()


class TestCheckConstraintEnumValues:
    def test_status_check_in_pattern(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        products = schema.get_table("products")
        assert products is not None
        status = products.get_column("status")
        assert status is not None
        assert status.enum_values is not None
        assert sorted(status.enum_values) == ["active", "archived", "draft"]

    def test_status_check_constraint_text(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        products = schema.get_table("products")
        assert products is not None
        status = products.get_column("status")
        assert status is not None
        assert status.check_constraint is not None
        assert "IN" in status.check_constraint


class TestNullableDefaults:
    def test_age_nullable_no_default(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        users = schema.get_table("users")
        assert users is not None
        age = users.get_column("age")
        assert age is not None
        assert age.nullable is True
        assert age.default is None

    def test_notes_nullable(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        orders = schema.get_table("orders")
        assert orders is not None
        notes = orders.get_column("notes")
        assert notes is not None
        assert notes.nullable is True


class TestDefaultValues:
    def test_is_active_default(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        users = schema.get_table("users")
        assert users is not None
        is_active = users.get_column("is_active")
        assert is_active is not None
        assert is_active.default is not None
        assert "1" in is_active.default

    def test_quantity_default(self, sqlite_url: str) -> None:
        schema = introspect(sqlite_url)
        oi = schema.get_table("order_items")
        assert oi is not None
        qty = oi.get_column("quantity")
        assert qty is not None
        assert qty.default is not None
        assert "1" in qty.default


# ── Coverage: additional branch tests ────────────────────────────────────


class TestGetRawTypeFallback:
    """Cover the exception branch in _get_raw_type."""

    def test_raw_type_from_compile(self, sqlite_url: str) -> None:
        assert _get_raw_type(sa.types.VARCHAR(100)) == "VARCHAR(100)"

    def test_raw_type_fallback_on_error(self) -> None:
        class BadType:
            def compile(self) -> None:
                raise TypeError("no compile")

            def __str__(self) -> str:
                return "BAD"

        assert _get_raw_type(BadType()) == "BAD"


class TestDetectAutoincrementExplicit:
    """Cover the col_info autoincrement=True branch."""

    def test_explicit_autoincrement_flag(self) -> None:
        col_info = {
            "name": "id",
            "type": sa.types.Integer(),
            "nullable": False,
            "default": None,
            "autoincrement": True,
        }
        assert _detect_autoincrement(col_info, True, ["id"], "postgresql") is True  # type: ignore[arg-type]

    def test_non_integer_pk_no_autoincrement(self) -> None:
        col_info = {
            "name": "id",
            "type": sa.types.VARCHAR(36),
            "nullable": False,
            "default": None,
        }
        assert _detect_autoincrement(col_info, True, ["id"], "sqlite") is False  # type: ignore[arg-type]


class TestExtractCheckEnumBranches:
    """Cover CHECK constraint edge cases."""

    def test_check_in_without_quoted_values(self) -> None:
        mock_inspector = MagicMock()
        mock_inspector.get_check_constraints.return_value = [{"sqltext": "status IN (1, 2, 3)"}]
        constraint, values = _extract_check_enum("t", "status", mock_inspector)
        assert constraint == "status IN (1, 2, 3)"
        assert values is None

    def test_check_constraint_exception(self) -> None:
        mock_inspector = MagicMock()
        mock_inspector.get_check_constraints.side_effect = NotImplementedError("unsupported")
        constraint, values = _extract_check_enum("t", "col", mock_inspector)
        assert constraint is None
        assert values is None


class TestCollectUniqueColumnsExplicit:
    """Cover the explicit unique constraints branch."""

    def test_explicit_unique_constraint(self) -> None:
        mock_inspector = MagicMock()
        mock_inspector.dialect.name = "postgresql"
        uniques = [{"column_names": ["email"]}]  # type: ignore[list-item]
        indexes: list[Any] = []
        result = _collect_unique_columns(uniques, indexes, "users", mock_inspector)
        assert "email" in result

    def test_multi_column_unique_ignored(self) -> None:
        mock_inspector = MagicMock()
        mock_inspector.dialect.name = "postgresql"
        uniques = [{"column_names": ["a", "b"]}]  # type: ignore[list-item]
        indexes: list[Any] = []
        result = _collect_unique_columns(uniques, indexes, "t", mock_inspector)
        assert len(result) == 0

    def test_unique_index_detected(self) -> None:
        mock_inspector = MagicMock()
        mock_inspector.dialect.name = "postgresql"
        uniques: list[Any] = []
        indexes = [{"unique": True, "column_names": ["code"]}]  # type: ignore[list-item]
        result = _collect_unique_columns(uniques, indexes, "t", mock_inspector)
        assert "code" in result


class TestSchemaReExports:
    """Verify the public API re-exports from dbsprout.schema."""

    def test_introspect_importable(self) -> None:
        assert callable(introspect_fn)

    def test_normalize_type_importable(self) -> None:
        assert callable(normalize_type_fn)


class TestSafeIdentifier:
    """Cover the _is_safe_identifier allowlist."""

    def test_normal_identifier(self) -> None:
        assert _is_safe_identifier("users") is True

    def test_autoindex_name(self) -> None:
        assert _is_safe_identifier("sqlite_autoindex_users_1") is True

    def test_identifier_with_double_quote(self) -> None:
        assert _is_safe_identifier('table"; DROP') is False

    def test_identifier_with_semicolon(self) -> None:
        assert _is_safe_identifier("table;evil") is False

    def test_identifier_with_null_byte(self) -> None:
        assert _is_safe_identifier("table\x00") is False

    def test_empty_string(self) -> None:
        assert _is_safe_identifier("") is False

    def test_starts_with_digit(self) -> None:
        assert _is_safe_identifier("1table") is False

    def test_too_long(self) -> None:
        assert _is_safe_identifier("a" * 129) is False

    def test_max_length_ok(self) -> None:
        assert _is_safe_identifier("a" * 128) is True


class TestValidateUrl:
    """Cover URL validation for supported dialects."""

    def test_sqlite_allowed(self) -> None:
        _validate_url("sqlite:///test.db")

    def test_postgresql_allowed(self) -> None:
        _validate_url("postgresql://user:pass@localhost/db")

    def test_mysql_allowed(self) -> None:
        _validate_url("mysql://user:pass@localhost/db")

    def test_unsupported_dialect_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dialect"):
            _validate_url("mssql://user:pass@localhost/db")


class TestConnectionTimeout:
    def test_connect_timeout_constant(self) -> None:
        assert _CONNECT_TIMEOUT == 10

    def test_pg_url_gets_connect_timeout(self) -> None:
        with patch("dbsprout.schema.introspect.sa.create_engine") as mock_create:
            mock_engine = MagicMock()
            mock_engine.dialect.name = "postgresql"
            mock_create.return_value = mock_engine
            _create_engine("postgresql://user:pass@localhost/db")
            _, kwargs = mock_create.call_args
            assert kwargs["connect_args"]["connect_timeout"] == 10

    def test_mysql_url_gets_connect_timeout(self) -> None:
        with patch("dbsprout.schema.introspect.sa.create_engine") as mock_create:
            mock_engine = MagicMock()
            mock_engine.dialect.name = "mysql"
            mock_create.return_value = mock_engine
            _create_engine("mysql+pymysql://user:pass@localhost/db")
            _, kwargs = mock_create.call_args
            assert kwargs["connect_args"]["connect_timeout"] == 10

    def test_sqlite_url_no_connect_timeout(self) -> None:
        with (
            patch("dbsprout.schema.introspect.sa.create_engine") as mock_create,
            patch("dbsprout.schema.introspect.event"),
        ):
            mock_engine = MagicMock()
            mock_engine.dialect.name = "sqlite"
            mock_create.return_value = mock_engine
            _create_engine("sqlite:///test.db")
            _, kwargs = mock_create.call_args
            assert kwargs.get("connect_args", {}) == {}


class TestCredentialSanitization:
    def test_password_not_in_error_message(self) -> None:
        with patch(
            "dbsprout.schema.introspect.sa.create_engine",
            side_effect=sa.exc.SQLAlchemyError("connection refused"),
        ):
            with pytest.raises(sa.exc.SQLAlchemyError, match="Failed to create engine") as exc_info:
                _create_engine("postgresql://admin:s3cret_p4ss@db.example.com/mydb")
            error_msg = str(exc_info.value)
            assert "s3cret_p4ss" not in error_msg
            assert "db.example.com" in error_msg
