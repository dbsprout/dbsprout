"""Tests for dbsprout.schema.parsers.ddl — SQL DDL file parsing."""

from __future__ import annotations

import pytest

from dbsprout.schema.models import ColumnType
from dbsprout.schema.parsers.ddl import parse_ddl

# ── Basic table parsing ──────────────────────────────────────────────────


class TestBasicCreateTable:
    def test_columns_extracted(self) -> None:
        ddl = "CREATE TABLE users (id INTEGER, name VARCHAR(100), active BOOLEAN);"
        schema = parse_ddl(ddl)
        users = schema.get_table("users")
        assert users is not None
        assert len(users.columns) == 3
        assert users.columns[0].name == "id"
        assert users.columns[1].name == "name"

    def test_column_types(self) -> None:
        ddl = "CREATE TABLE t (a INTEGER, b VARCHAR(255), c TEXT, d BOOLEAN, e DECIMAL(10,2));"
        schema = parse_ddl(ddl)
        t = schema.get_table("t")
        assert t is not None
        assert t.columns[0].data_type is ColumnType.INTEGER
        assert t.columns[1].data_type is ColumnType.VARCHAR
        assert t.columns[1].max_length == 255
        assert t.columns[2].data_type is ColumnType.TEXT
        assert t.columns[3].data_type is ColumnType.BOOLEAN
        assert t.columns[4].data_type is ColumnType.DECIMAL
        assert t.columns[4].precision == 10
        assert t.columns[4].scale == 2


# ── Primary keys ─────────────────────────────────────────────────────────


class TestPrimaryKey:
    def test_inline_primary_key(self) -> None:
        ddl = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"
        schema = parse_ddl(ddl)
        users = schema.get_table("users")
        assert users is not None
        assert users.primary_key == ["id"]

    def test_table_level_primary_key(self) -> None:
        ddl = """CREATE TABLE order_items (
            order_id INTEGER, product_id INTEGER,
            PRIMARY KEY (order_id, product_id)
        );"""
        schema = parse_ddl(ddl)
        oi = schema.get_table("order_items")
        assert oi is not None
        assert sorted(oi.primary_key) == ["order_id", "product_id"]


# ── Constraints ──────────────────────────────────────────────────────────


class TestConstraints:
    def test_not_null(self) -> None:
        ddl = "CREATE TABLE t (id INTEGER NOT NULL, name TEXT);"
        schema = parse_ddl(ddl)
        t = schema.get_table("t")
        assert t is not None
        assert t.columns[0].nullable is False
        assert t.columns[1].nullable is True

    def test_unique(self) -> None:
        ddl = "CREATE TABLE t (id INTEGER, email VARCHAR(255) UNIQUE);"
        schema = parse_ddl(ddl)
        t = schema.get_table("t")
        assert t is not None
        assert t.columns[1].unique is True

    def test_default_value(self) -> None:
        ddl = "CREATE TABLE t (id INTEGER, status VARCHAR(20) DEFAULT 'active');"
        schema = parse_ddl(ddl)
        t = schema.get_table("t")
        assert t is not None
        assert t.columns[1].default is not None
        assert "active" in t.columns[1].default


# ── Foreign keys ─────────────────────────────────────────────────────────


class TestForeignKeys:
    def test_inline_fk(self) -> None:
        ddl = """
        CREATE TABLE users (id INTEGER PRIMARY KEY);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id));
        """
        schema = parse_ddl(ddl)
        orders = schema.get_table("orders")
        assert orders is not None
        assert len(orders.foreign_keys) == 1
        assert orders.foreign_keys[0].ref_table == "users"
        assert orders.foreign_keys[0].columns == ["user_id"]

    def test_table_level_fk(self) -> None:
        ddl = """
        CREATE TABLE users (id INTEGER PRIMARY KEY);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """
        schema = parse_ddl(ddl)
        orders = schema.get_table("orders")
        assert orders is not None
        assert len(orders.foreign_keys) == 1
        assert orders.foreign_keys[0].ref_table == "users"

    def test_alter_table_add_fk(self) -> None:
        ddl = """
        CREATE TABLE users (id INTEGER PRIMARY KEY);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER);
        ALTER TABLE orders ADD FOREIGN KEY (user_id) REFERENCES users(id);
        """
        schema = parse_ddl(ddl)
        orders = schema.get_table("orders")
        assert orders is not None
        assert len(orders.foreign_keys) == 1

    def test_on_delete_cascade(self) -> None:
        ddl = """
        CREATE TABLE users (id INTEGER PRIMARY KEY);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY, user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
        schema = parse_ddl(ddl)
        orders = schema.get_table("orders")
        assert orders is not None
        assert orders.foreign_keys[0].on_delete == "CASCADE"


# ── Indexes ──────────────────────────────────────────────────────────────


class TestIndexes:
    def test_create_index(self) -> None:
        ddl = """
        CREATE TABLE users (id INTEGER PRIMARY KEY, email VARCHAR(255));
        CREATE INDEX idx_email ON users (email);
        """
        schema = parse_ddl(ddl)
        users = schema.get_table("users")
        assert users is not None
        assert len(users.indexes) >= 1
        idx = next(i for i in users.indexes if i.name == "idx_email")
        assert idx.columns == ["email"]
        assert idx.unique is False

    def test_create_unique_index(self) -> None:
        ddl = """
        CREATE TABLE users (id INTEGER PRIMARY KEY, email VARCHAR(255));
        CREATE UNIQUE INDEX idx_email ON users (email);
        """
        schema = parse_ddl(ddl)
        users = schema.get_table("users")
        assert users is not None
        idx = next(i for i in users.indexes if i.name == "idx_email")
        assert idx.unique is True


# ── Dialect-specific ─────────────────────────────────────────────────────


class TestDialects:
    def test_serial_postgres(self) -> None:
        ddl = "CREATE TABLE users (id SERIAL PRIMARY KEY, name TEXT);"
        schema = parse_ddl(ddl, dialect="postgres")
        users = schema.get_table("users")
        assert users is not None
        assert users.columns[0].autoincrement is True
        assert users.columns[0].data_type in (ColumnType.INTEGER, ColumnType.BIGINT)

    def test_auto_increment_mysql(self) -> None:
        ddl = "CREATE TABLE users (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(100));"
        schema = parse_ddl(ddl, dialect="mysql")
        users = schema.get_table("users")
        assert users is not None
        assert users.columns[0].autoincrement is True

    def test_dialect_auto_detection_serial(self) -> None:
        ddl = "CREATE TABLE users (id SERIAL PRIMARY KEY, name TEXT);"
        schema = parse_ddl(ddl)
        assert schema.dialect == "postgresql"

    def test_dialect_auto_detection_auto_increment(self) -> None:
        ddl = "CREATE TABLE users (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(100));"
        schema = parse_ddl(ddl)
        assert schema.dialect == "mysql"

    def test_explicit_dialect_override(self) -> None:
        ddl = "CREATE TABLE t (id INTEGER PRIMARY KEY);"
        schema = parse_ddl(ddl, dialect="postgres")
        assert schema.dialect == "postgresql"


# ── Metadata ─────────────────────────────────────────────────────────────


class TestMetadata:
    def test_source_is_ddl(self) -> None:
        ddl = "CREATE TABLE t (id INTEGER);"
        schema = parse_ddl(ddl)
        assert schema.source == "ddl"

    def test_source_file_set(self) -> None:
        ddl = "CREATE TABLE t (id INTEGER);"
        schema = parse_ddl(ddl, source_file="schema.sql")
        assert schema.source_file == "schema.sql"

    def test_table_count(self) -> None:
        ddl = """
        CREATE TABLE a (id INTEGER);
        CREATE TABLE b (id INTEGER);
        CREATE TABLE c (id INTEGER);
        """
        schema = parse_ddl(ddl)
        assert len(schema.tables) == 3


# ── Edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match=r"No.*statement"):
            parse_ddl("")

    def test_comment_only_raises(self) -> None:
        with pytest.raises(ValueError, match=r"No.*statement"):
            parse_ddl("-- just a comment")

    def test_create_table_if_not_exists(self) -> None:
        ddl = "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT);"
        schema = parse_ddl(ddl)
        assert schema.get_table("users") is not None

    def test_raw_type_preserved(self) -> None:
        ddl = "CREATE TABLE t (price DECIMAL(10,2), name VARCHAR(100));"
        schema = parse_ddl(ddl)
        t = schema.get_table("t")
        assert t is not None
        assert t.columns[0].raw_type != ""
        assert t.columns[1].raw_type != ""


# ── Multi-table realistic DDL ────────────────────────────────────────────


class TestRealisticDDL:
    def test_sqlite_multi_table(self) -> None:
        ddl = """
        CREATE TABLE categories (
            id INTEGER PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            parent_id INTEGER REFERENCES categories(id)
        );
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name VARCHAR(200) NOT NULL,
            price DECIMAL(10, 2) NOT NULL,
            category_id INTEGER REFERENCES categories(id)
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            product_id INTEGER,
            quantity INTEGER DEFAULT 1,
            FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_orders_product ON orders (product_id);
        """
        schema = parse_ddl(ddl)
        assert len(schema.tables) == 3
        assert schema.get_table("categories") is not None
        assert schema.get_table("products") is not None
        assert schema.get_table("orders") is not None

        orders = schema.get_table("orders")
        assert orders is not None
        assert len(orders.foreign_keys) == 1
        assert orders.foreign_keys[0].on_delete == "CASCADE"

        cats = schema.get_table("categories")
        assert cats is not None
        # Self-referencing FK
        self_fks = [fk for fk in cats.foreign_keys if fk.ref_table == "categories"]
        assert len(self_fks) == 1


# ── Additional coverage tests ────────────────────────────────────────────


class TestTypeMapping:
    """Cover type mapping paths in _map_sqlglot_type."""

    def test_bigint(self) -> None:
        schema = parse_ddl("CREATE TABLE t (id BIGINT);")
        assert schema.get_table("t") is not None
        assert schema.get_table("t").columns[0].data_type is ColumnType.BIGINT

    def test_smallint(self) -> None:
        schema = parse_ddl("CREATE TABLE t (id SMALLINT);")
        assert schema.get_table("t").columns[0].data_type is ColumnType.SMALLINT

    def test_float(self) -> None:
        schema = parse_ddl("CREATE TABLE t (v FLOAT);")
        assert schema.get_table("t").columns[0].data_type is ColumnType.FLOAT

    def test_double(self) -> None:
        schema = parse_ddl("CREATE TABLE t (v DOUBLE);")
        assert schema.get_table("t").columns[0].data_type is ColumnType.FLOAT

    def test_date(self) -> None:
        schema = parse_ddl("CREATE TABLE t (d DATE);")
        assert schema.get_table("t").columns[0].data_type is ColumnType.DATE

    def test_datetime(self) -> None:
        schema = parse_ddl("CREATE TABLE t (d DATETIME);")
        assert schema.get_table("t").columns[0].data_type is ColumnType.DATETIME

    def test_timestamp(self) -> None:
        schema = parse_ddl("CREATE TABLE t (d TIMESTAMP);")
        assert schema.get_table("t").columns[0].data_type is ColumnType.TIMESTAMP

    def test_time(self) -> None:
        schema = parse_ddl("CREATE TABLE t (d TIME);")
        assert schema.get_table("t").columns[0].data_type is ColumnType.TIME

    def test_uuid(self) -> None:
        schema = parse_ddl("CREATE TABLE t (id UUID);", dialect="postgres")
        assert schema.get_table("t").columns[0].data_type is ColumnType.UUID

    def test_json(self) -> None:
        schema = parse_ddl("CREATE TABLE t (data JSON);")
        assert schema.get_table("t").columns[0].data_type is ColumnType.JSON

    def test_jsonb(self) -> None:
        schema = parse_ddl("CREATE TABLE t (data JSONB);", dialect="postgres")
        assert schema.get_table("t").columns[0].data_type is ColumnType.JSON

    def test_binary(self) -> None:
        schema = parse_ddl("CREATE TABLE t (data BINARY);")
        assert schema.get_table("t").columns[0].data_type is ColumnType.BINARY

    def test_blob(self) -> None:
        schema = parse_ddl("CREATE TABLE t (data BLOB);")
        assert schema.get_table("t").columns[0].data_type is ColumnType.BINARY

    def test_unknown_type(self) -> None:
        schema = parse_ddl("CREATE TABLE t (x CUSTOMTYPE);")
        assert schema.get_table("t").columns[0].data_type is ColumnType.UNKNOWN

    def test_char(self) -> None:
        schema = parse_ddl("CREATE TABLE t (c CHAR(10));")
        assert schema.get_table("t").columns[0].data_type is ColumnType.VARCHAR

    def test_bigserial(self) -> None:
        schema = parse_ddl("CREATE TABLE t (id BIGSERIAL);", dialect="postgres")
        t = schema.get_table("t")
        assert t.columns[0].data_type is ColumnType.BIGINT
        assert t.columns[0].autoincrement is True

    def test_mediumtext(self) -> None:
        schema = parse_ddl("CREATE TABLE t (body MEDIUMTEXT);", dialect="mysql")
        assert schema.get_table("t").columns[0].data_type is ColumnType.TEXT

    def test_varbinary(self) -> None:
        schema = parse_ddl("CREATE TABLE t (data VARBINARY(100));")
        assert schema.get_table("t").columns[0].data_type is ColumnType.BINARY


class TestCheckConstraint:
    def test_check_in_pattern(self) -> None:
        ddl = "CREATE TABLE t (status VARCHAR(20) CHECK (status IN ('a', 'b', 'c')));"
        schema = parse_ddl(ddl)
        t = schema.get_table("t")
        assert t.columns[0].enum_values == ["a", "b", "c"]
        assert t.columns[0].data_type is ColumnType.ENUM

    def test_check_no_in_pattern(self) -> None:
        ddl = "CREATE TABLE t (age INT CHECK (age > 0));"
        schema = parse_ddl(ddl)
        t = schema.get_table("t")
        assert t.columns[0].check_constraint is not None
        assert t.columns[0].enum_values is None


class TestAlterTableEdgeCases:
    def test_alter_on_nonexistent_table_ignored(self) -> None:
        ddl = """
        CREATE TABLE users (id INTEGER PRIMARY KEY);
        ALTER TABLE orders ADD FOREIGN KEY (user_id) REFERENCES users(id);
        """
        schema = parse_ddl(ddl)
        assert len(schema.tables) == 1  # orders not created, ALTER ignored

    def test_named_fk_constraint(self) -> None:
        ddl = """
        CREATE TABLE users (id INTEGER PRIMARY KEY);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER,
            CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id));
        """
        schema = parse_ddl(ddl)
        orders = schema.get_table("orders")
        assert len(orders.foreign_keys) == 1


class TestDialectDetectionEdgeCases:
    def test_backtick_detection(self) -> None:
        ddl = "CREATE TABLE `users` (`id` INT AUTO_INCREMENT PRIMARY KEY);"
        schema = parse_ddl(ddl)
        assert schema.dialect == "mysql"

    def test_sqlite_autoincrement(self) -> None:
        ddl = "CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT);"
        schema = parse_ddl(ddl, dialect="sqlite")
        assert schema.dialect == "sqlite"

    def test_no_dialect_detected(self) -> None:
        ddl = "CREATE TABLE t (id INTEGER PRIMARY KEY);"
        schema = parse_ddl(ddl)
        assert schema.dialect is None


class TestMultiColumnFK:
    def test_composite_fk(self) -> None:
        ddl = """
        CREATE TABLE parent (a INT, b INT, PRIMARY KEY (a, b));
        CREATE TABLE child (id INT PRIMARY KEY, a INT, b INT,
            FOREIGN KEY (a, b) REFERENCES parent(a, b));
        """
        schema = parse_ddl(ddl)
        child = schema.get_table("child")
        assert len(child.foreign_keys) == 1
        assert child.foreign_keys[0].columns == ["a", "b"]
        assert child.foreign_keys[0].ref_columns == ["a", "b"]


class TestMultipleStatementTypes:
    def test_mixed_statements(self) -> None:
        """Non-CREATE/ALTER statements are silently ignored."""
        ddl = """
        CREATE TABLE users (id INTEGER PRIMARY KEY);
        INSERT INTO users VALUES (1);
        SELECT * FROM users;
        DROP TABLE IF EXISTS temp;
        """
        schema = parse_ddl(ddl)
        assert len(schema.tables) == 1

    def test_on_update_fk(self) -> None:
        ddl = """
        CREATE TABLE p (id INT PRIMARY KEY);
        CREATE TABLE c (id INT PRIMARY KEY, p_id INT,
            FOREIGN KEY (p_id) REFERENCES p(id) ON UPDATE CASCADE ON DELETE SET NULL);
        """
        schema = parse_ddl(ddl)
        c = schema.get_table("c")
        fk = c.foreign_keys[0]
        assert fk.on_update == "CASCADE"
        assert fk.on_delete == "SET NULL"


class TestEnumType:
    def test_pg_enum_column(self) -> None:
        ddl = "CREATE TABLE t (status ENUM('active', 'inactive'));"
        schema = parse_ddl(ddl)
        t = schema.get_table("t")
        assert t.columns[0].data_type is ColumnType.ENUM


class TestDecimalNoPrecision:
    def test_decimal_without_params(self) -> None:
        ddl = "CREATE TABLE t (amount DECIMAL);"
        schema = parse_ddl(ddl)
        t = schema.get_table("t")
        assert t.columns[0].data_type is ColumnType.DECIMAL
        assert t.columns[0].precision is None

    def test_decimal_with_precision_only(self) -> None:
        ddl = "CREATE TABLE t (amount DECIMAL(10));"
        schema = parse_ddl(ddl)
        t = schema.get_table("t")
        assert t.columns[0].precision == 10
        assert t.columns[0].scale is None


class TestTableLevelUniqueConstraint:
    def test_unique_constraint(self) -> None:
        ddl = """CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            email VARCHAR(255),
            UNIQUE (email)
        );"""
        schema = parse_ddl(ddl)
        users = schema.get_table("users")
        email = users.get_column("email")
        assert email is not None
        assert email.unique is True
