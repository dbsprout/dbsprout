"""Integration test fixtures — ephemeral PostgreSQL and MySQL containers.

Requires Docker. Tests auto-skip when Docker is unavailable.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

# Auto-skip the entire integration directory when testcontainers is missing
tc = pytest.importorskip("testcontainers", reason="testcontainers not installed")


def _docker_available() -> bool:
    """Check if Docker daemon is reachable."""
    try:
        import docker  # type: ignore[import-untyped]  # noqa: PLC0415

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


if not _docker_available():
    pytest.skip("Docker daemon not available", allow_module_level=True)

from testcontainers.mysql import MySqlContainer  # type: ignore[import-untyped]  # noqa: E402
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]  # noqa: E402


def _col(
    name: str,
    dtype: ColumnType = ColumnType.INTEGER,
    **kwargs: Any,
) -> ColumnSchema:
    return ColumnSchema(name=name, data_type=dtype, **kwargs)


@pytest.fixture(scope="session")
def pg_container() -> Any:
    """Session-scoped PostgreSQL container."""
    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


@pytest.fixture(scope="session")
def pg_url(pg_container: Any) -> str:
    """PostgreSQL connection URL (psycopg3 format)."""
    # testcontainers gives psycopg2 URL; convert to psycopg3 format
    url = pg_container.get_connection_url()
    return url.replace("psycopg2", "psycopg").replace("postgresql+psycopg", "postgresql")


@pytest.fixture(scope="session")
def mysql_container() -> Any:
    """Session-scoped MySQL container with local_infile enabled."""
    with MySqlContainer("mysql:8.0") as mysql:
        # Enable local_infile on server
        import pymysql  # type: ignore[import-untyped]  # noqa: PLC0415

        params = {
            "host": mysql.get_container_host_ip(),
            "port": int(mysql.get_exposed_port(3306)),
            "user": "root",
            "password": mysql.root_password if hasattr(mysql, "root_password") else "test",
            "database": mysql.dbname if hasattr(mysql, "dbname") else "test",
            "local_infile": True,
        }
        conn = pymysql.connect(**params)
        try:
            with conn.cursor() as cur:
                cur.execute("SET GLOBAL local_infile = 1")
            conn.commit()
        finally:
            conn.close()
        yield mysql


@pytest.fixture(scope="session")
def mysql_url(mysql_container: Any) -> str:
    """MySQL connection URL."""
    url = mysql_container.get_connection_url()
    # Strip driver prefix (e.g., mysql+pymysql://) → mysql://
    if "+pymysql" in url:
        url = url.replace("+pymysql", "")
    elif "+mysqldb" in url:
        url = url.replace("+mysqldb", "")
    return url


@pytest.fixture
def test_schema() -> DatabaseSchema:
    """Two tables: users (PK serial) → posts (FK to users)."""
    users = TableSchema(
        name="users",
        columns=[
            _col("id", primary_key=True, autoincrement=True, nullable=False),
            _col("email", ColumnType.VARCHAR, max_length=255, nullable=False, unique=True),
        ],
        primary_key=["id"],
    )
    posts = TableSchema(
        name="posts",
        columns=[
            _col("id", primary_key=True, autoincrement=True, nullable=False),
            _col("user_id", nullable=False),
            _col("title", ColumnType.VARCHAR, max_length=255, nullable=False),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(
                columns=["user_id"],
                ref_table="users",
                ref_columns=["id"],
                on_delete="CASCADE",
            )
        ],
    )
    return DatabaseSchema(tables=[users, posts])


@pytest.fixture
def test_rows() -> dict[str, list[dict[str, Any]]]:
    """100 users + 500 posts with valid FK refs."""
    rng = random.Random(42)  # noqa: S311
    users = [{"id": i + 1, "email": f"user{i + 1}@test.com"} for i in range(100)]
    posts = [
        {
            "id": i + 1,
            "user_id": rng.randint(1, 100),
            "title": f"Post {i + 1}",
        }
        for i in range(500)
    ]
    return {"users": users, "posts": posts}


def create_pg_tables(pg_url: str, schema: DatabaseSchema) -> None:
    """Create tables in PostgreSQL from a DatabaseSchema."""
    import psycopg  # type: ignore[import-not-found]  # noqa: PLC0415

    with psycopg.connect(pg_url) as conn, conn.cursor() as cur:
        for table in schema.tables:
            cur.execute(f'DROP TABLE IF EXISTS "{table.name}" CASCADE')
        # Create users first, then posts (respecting FK order)
        for table in schema.tables:
            cols = []
            for col in table.columns:
                parts = [f'"{col.name}"']
                if col.autoincrement:
                    parts.append("SERIAL")
                elif col.data_type == ColumnType.VARCHAR:
                    parts.append(f"VARCHAR({col.max_length or 255})")
                else:
                    parts.append("INTEGER")
                if not col.nullable:
                    parts.append("NOT NULL")
                if col.unique:
                    parts.append("UNIQUE")
                cols.append(" ".join(parts))
            if len(table.primary_key) > 0:
                pk = ", ".join(f'"{c}"' for c in table.primary_key)
                cols.append(f"PRIMARY KEY ({pk})")
            for fk in table.foreign_keys:
                fk_cols = ", ".join(f'"{c}"' for c in fk.columns)
                ref_cols = ", ".join(f'"{c}"' for c in fk.ref_columns)
                cols.append(f'FOREIGN KEY ({fk_cols}) REFERENCES "{fk.ref_table}" ({ref_cols})')
            ddl = f'CREATE TABLE "{table.name}" ({", ".join(cols)})'
            cur.execute(ddl)
        conn.commit()


def drop_pg_tables(pg_url: str, schema: DatabaseSchema) -> None:
    """Drop tables in PostgreSQL."""
    import psycopg  # type: ignore[import-not-found]  # noqa: PLC0415

    with psycopg.connect(pg_url) as conn, conn.cursor() as cur:
        for table in reversed(schema.tables):
            cur.execute(f'DROP TABLE IF EXISTS "{table.name}" CASCADE')
        conn.commit()


def create_mysql_tables(mysql_url: str, schema: DatabaseSchema) -> None:
    """Create tables in MySQL from a DatabaseSchema."""
    import pymysql  # type: ignore[import-untyped]  # noqa: PLC0415

    from dbsprout.output.mysql_load_data import _parse_mysql_url  # noqa: PLC0415

    params = _parse_mysql_url(mysql_url)
    conn = pymysql.connect(**params)
    try:
        with conn.cursor() as cur:
            cur.execute("SET FOREIGN_KEY_CHECKS=0")
            for table in schema.tables:
                cur.execute(f"DROP TABLE IF EXISTS `{table.name}`")
            cur.execute("SET FOREIGN_KEY_CHECKS=1")
            for table in schema.tables:
                cols = []
                for col in table.columns:
                    parts = [f"`{col.name}`"]
                    if col.autoincrement:
                        parts.append("INT AUTO_INCREMENT")
                    elif col.data_type == ColumnType.VARCHAR:
                        parts.append(f"VARCHAR({col.max_length or 255})")
                    else:
                        parts.append("INT")
                    if not col.nullable:
                        parts.append("NOT NULL")
                    if col.unique:
                        parts.append("UNIQUE")
                    cols.append(" ".join(parts))
                if len(table.primary_key) > 0:
                    pk = ", ".join(f"`{c}`" for c in table.primary_key)
                    cols.append(f"PRIMARY KEY ({pk})")
                for fk in table.foreign_keys:
                    fk_cols = ", ".join(f"`{c}`" for c in fk.columns)
                    ref_cols = ", ".join(f"`{c}`" for c in fk.ref_columns)
                    cols.append(f"FOREIGN KEY ({fk_cols}) REFERENCES `{fk.ref_table}` ({ref_cols})")
                ddl = f"CREATE TABLE `{table.name}` ({', '.join(cols)}) ENGINE=InnoDB"
                cur.execute(ddl)
        conn.commit()
    finally:
        conn.close()


def drop_mysql_tables(mysql_url: str, schema: DatabaseSchema) -> None:
    """Drop tables in MySQL."""
    import pymysql  # type: ignore[import-untyped]  # noqa: PLC0415

    from dbsprout.output.mysql_load_data import _parse_mysql_url  # noqa: PLC0415

    params = _parse_mysql_url(mysql_url)
    conn = pymysql.connect(**params)
    try:
        with conn.cursor() as cur:
            cur.execute("SET FOREIGN_KEY_CHECKS=0")
            for table in reversed(schema.tables):
                cur.execute(f"DROP TABLE IF EXISTS `{table.name}`")
            cur.execute("SET FOREIGN_KEY_CHECKS=1")
        conn.commit()
    finally:
        conn.close()
