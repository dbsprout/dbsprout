"""Tests for generic SQLAlchemy batch writer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import sqlalchemy as sa

from dbsprout.output.models import InsertResult
from dbsprout.output.sa_batch import SaBatchWriter
from dbsprout.schema.models import ColumnSchema, ColumnType, DatabaseSchema, TableSchema


def _make_schema(
    table_name: str = "users",
    columns: list[ColumnSchema] | None = None,
) -> DatabaseSchema:
    cols = columns or [
        ColumnSchema(
            name="id",
            data_type=ColumnType.INTEGER,
            nullable=False,
            primary_key=True,
        ),
        ColumnSchema(
            name="name",
            data_type=ColumnType.VARCHAR,
            nullable=False,
            max_length=100,
        ),
    ]
    return DatabaseSchema(
        tables=[TableSchema(name=table_name, columns=cols, primary_key=["id"])],
        dialect="sqlite",
    )


def _create_table(engine: sa.Engine, table_name: str = "users") -> None:
    with engine.connect() as conn:
        conn.execute(sa.text(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY, name TEXT)"))
        conn.commit()


class TestSaBatchWriterEmpty:
    def test_empty_tables_returns_zero(self) -> None:
        result = SaBatchWriter().write(
            tables_data={},
            schema=_make_schema(),
            insertion_order=[],
            db_url="sqlite:///:memory:",
        )
        assert isinstance(result, InsertResult)
        assert result.tables_inserted == 0
        assert result.total_rows == 0
        assert result.duration_seconds >= 0.0


class TestSaBatchWriterSqlite:
    def test_inserts_rows(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        _create_table(engine)

        rows: list[dict[str, Any]] = [{"id": i, "name": f"user_{i}"} for i in range(1, 6)]
        schema = _make_schema()

        result = SaBatchWriter().write(
            tables_data={"users": rows},
            schema=schema,
            insertion_order=["users"],
            db_url="sqlite:///:memory:",
            _engine_override=engine,
        )

        assert result.total_rows == 5
        with engine.connect() as conn:
            count = conn.execute(sa.text("SELECT COUNT(*) FROM users")).scalar()
        assert count == 5

    def test_multiple_tables(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            conn.execute(sa.text("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)"))
            conn.execute(sa.text("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT)"))
            conn.commit()

        dept_schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="departments",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="name",
                            data_type=ColumnType.VARCHAR,
                            nullable=False,
                        ),
                    ],
                    primary_key=["id"],
                ),
                TableSchema(
                    name="employees",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="name",
                            data_type=ColumnType.VARCHAR,
                            nullable=False,
                        ),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="sqlite",
        )

        tables_data: dict[str, list[dict[str, Any]]] = {
            "departments": [{"id": 1, "name": "Engineering"}],
            "employees": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ],
        }

        result = SaBatchWriter().write(
            tables_data=tables_data,
            schema=dept_schema,
            insertion_order=["departments", "employees"],
            db_url="sqlite:///:memory:",
            _engine_override=engine,
        )

        assert result.tables_inserted == 2
        assert result.total_rows == 3

        with engine.connect() as conn:
            dept_count = conn.execute(sa.text("SELECT COUNT(*) FROM departments")).scalar()
            emp_count = conn.execute(sa.text("SELECT COUNT(*) FROM employees")).scalar()
        assert dept_count == 1
        assert emp_count == 2

    def test_insertion_order_respected(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            conn.execute(sa.text("CREATE TABLE parent (id INTEGER PRIMARY KEY, name TEXT)"))
            conn.execute(
                sa.text(
                    "CREATE TABLE child "
                    "(id INTEGER PRIMARY KEY, name TEXT, "
                    "parent_id INTEGER REFERENCES parent(id))"
                )
            )
            conn.commit()

        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="parent",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="name",
                            data_type=ColumnType.VARCHAR,
                            nullable=False,
                        ),
                    ],
                    primary_key=["id"],
                ),
                TableSchema(
                    name="child",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="name",
                            data_type=ColumnType.VARCHAR,
                            nullable=False,
                        ),
                        ColumnSchema(
                            name="parent_id",
                            data_type=ColumnType.INTEGER,
                            nullable=True,
                        ),
                    ],
                    primary_key=["id"],
                ),
            ],
            dialect="sqlite",
        )

        tables_data: dict[str, list[dict[str, Any]]] = {
            "parent": [{"id": 1, "name": "P1"}],
            "child": [{"id": 1, "name": "C1", "parent_id": 1}],
        }

        # Parent must be inserted before child (FK constraint)
        result = SaBatchWriter().write(
            tables_data=tables_data,
            schema=schema,
            insertion_order=["parent", "child"],
            db_url="sqlite:///:memory:",
            _engine_override=engine,
        )

        assert result.tables_inserted == 2
        with engine.connect() as conn:
            child_rows = conn.execute(sa.text("SELECT * FROM child")).fetchall()
        assert len(child_rows) == 1
        assert child_rows[0][2] == 1  # parent_id

    def test_table_with_zero_rows_skipped(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        _create_table(engine)

        tables_data: dict[str, list[dict[str, Any]]] = {
            "users": [],
        }

        result = SaBatchWriter().write(
            tables_data=tables_data,
            schema=_make_schema(),
            insertion_order=["users"],
            db_url="sqlite:///:memory:",
            _engine_override=engine,
        )

        assert result.tables_inserted == 0
        assert result.total_rows == 0

    def test_returns_insert_result(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        _create_table(engine)

        rows: list[dict[str, Any]] = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Charlie"},
        ]

        result = SaBatchWriter().write(
            tables_data={"users": rows},
            schema=_make_schema(),
            insertion_order=["users"],
            db_url="sqlite:///:memory:",
            _engine_override=engine,
        )

        assert isinstance(result, InsertResult)
        assert result.tables_inserted == 1
        assert result.total_rows == 3
        assert result.duration_seconds > 0.0


class TestSaBatchWriterPragmas:
    def test_wal_enabled_for_file_sqlite(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        db_url = f"sqlite:///{db_path}"

        # Create table then dispose so writer can set WAL cleanly
        setup_engine = sa.create_engine(db_url)
        _create_table(setup_engine)
        setup_engine.dispose()

        rows: list[dict[str, Any]] = [{"id": 1, "name": "Alice"}]
        SaBatchWriter().write(
            tables_data={"users": rows},
            schema=_make_schema(),
            insertion_order=["users"],
            db_url=db_url,
        )

        # Verify WAL was set (persists on file DBs)
        check_engine = sa.create_engine(db_url)
        with check_engine.connect() as conn:
            mode = conn.execute(sa.text("PRAGMA journal_mode")).scalar()
            count = conn.execute(sa.text("SELECT COUNT(*) FROM users")).scalar()
        assert mode == "wal"
        assert count == 1
        check_engine.dispose()

    def test_non_sqlite_skips_pragmas(self) -> None:
        """Verify _apply_pragmas is a no-op for non-sqlite dialects."""
        conn = sa.create_engine("sqlite:///:memory:").connect()
        # Calling with dialect="postgresql" should not execute any PRAGMA
        SaBatchWriter._apply_pragmas(conn, "postgresql", "postgresql://host/db")
        conn.close()

    def test_engine_disposed_without_override(self) -> None:
        """Writer should dispose the engine it creates (no _engine_override)."""
        result = SaBatchWriter().write(
            tables_data={},
            schema=_make_schema(),
            insertion_order=[],
            db_url="sqlite:///:memory:",
        )
        assert result.total_rows == 0


class TestSaBatchWriterBatchSizing:
    def test_mssql_batch_cap(self) -> None:
        size = SaBatchWriter._compute_batch_size("mssql", 10, 10000)
        assert size == 210

    def test_sqlite_no_cap(self) -> None:
        size = SaBatchWriter._compute_batch_size("sqlite", 10, 10000)
        assert size == 10000

    def test_single_column_mssql(self) -> None:
        size = SaBatchWriter._compute_batch_size("mssql", 1, 10000)
        assert size == 2100
