"""Database introspection via SQLAlchemy Inspector.

Connects to a live database, reads its schema, and returns a unified
``DatabaseSchema``. Currently supports SQLite; PostgreSQL and MySQL
introspection will be added in S-004 / S-005.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import event, inspect

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine, Inspector
    from sqlalchemy.engine.interfaces import (
        ReflectedColumn,
        ReflectedForeignKeyConstraint,
        ReflectedIndex,
        ReflectedUniqueConstraint,
    )

from dbsprout.schema.dialect import normalize_type
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    IndexSchema,
    TableSchema,
)

_SUPPORTED_DIALECTS = frozenset({"sqlite", "postgresql", "mysql"})


def introspect(url: str) -> DatabaseSchema:
    """Introspect a database and return a ``DatabaseSchema``.

    Parameters
    ----------
    url:
        SQLAlchemy-compatible database URL
        (e.g. ``"sqlite:///path/to/db.sqlite"``).

    Returns
    -------
    DatabaseSchema
        The complete, immutable schema representation.

    Raises
    ------
    ValueError
        If the URL uses an unsupported database dialect.
    """
    _validate_url(url)
    engine = _create_engine(url)
    try:
        inspector = inspect(engine)
        dialect_name = engine.dialect.name
        tables = _introspect_tables(inspector, dialect_name)
        return DatabaseSchema(
            tables=tables,
            dialect=dialect_name,
            source="introspect",
        )
    finally:
        engine.dispose()


# ── Private helpers ──────────────────────────────────────────────────────


def _validate_url(url: str) -> None:
    """Reject URLs with unsupported or dangerous dialects."""
    dialect = sa.engine.make_url(url).get_backend_name()
    if dialect not in _SUPPORTED_DIALECTS:
        msg = (
            f"Unsupported dialect {dialect!r}. Supported: {', '.join(sorted(_SUPPORTED_DIALECTS))}"
        )
        raise ValueError(msg)


def _create_engine(url: str) -> Engine:
    """Create a SQLAlchemy engine with dialect-specific setup."""
    engine = sa.create_engine(url)
    if engine.dialect.name == "sqlite":
        _register_sqlite_pragma(engine)
    return engine


def _register_sqlite_pragma(engine: Engine) -> None:
    """Enable ``PRAGMA foreign_keys = ON`` for every SQLite connection."""

    @event.listens_for(engine, "connect")
    def _set_sqlite_fk_pragma(
        dbapi_connection: Any,
        connection_record: Any,  # noqa: ARG001
    ) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()


def _introspect_tables(inspector: Inspector, dialect: str) -> list[TableSchema]:
    """Introspect all tables and return a list of ``TableSchema``."""
    table_names = inspector.get_table_names()
    return [_introspect_table(inspector, name, dialect) for name in table_names]


def _introspect_table(
    inspector: Inspector,
    table_name: str,
    dialect: str,
) -> TableSchema:
    """Introspect a single table and build a ``TableSchema``."""
    # Columns
    raw_columns = inspector.get_columns(table_name)
    pk_constraint = inspector.get_pk_constraint(table_name)
    pk_cols: list[str] = pk_constraint.get("constrained_columns", []) if pk_constraint else []

    columns = _build_columns(raw_columns, pk_cols, dialect, table_name, inspector)

    # Foreign keys
    raw_fks = inspector.get_foreign_keys(table_name)
    foreign_keys = _build_foreign_keys(raw_fks)

    # Unique constraints (explicit + from unique indexes + SQLite autoindexes)
    raw_uniques = inspector.get_unique_constraints(table_name)
    raw_indexes = inspector.get_indexes(table_name)
    unique_cols = _collect_unique_columns(raw_uniques, raw_indexes, table_name, inspector)

    # Mark unique columns
    columns = _apply_unique_flags(columns, unique_cols)

    # Indexes (exclude auto-created unique indexes for inline UNIQUE columns)
    indexes = _build_indexes(raw_indexes)

    return TableSchema(
        name=table_name,
        columns=columns,
        primary_key=pk_cols,
        foreign_keys=foreign_keys,
        indexes=indexes,
    )


def _build_columns(
    raw_columns: list[ReflectedColumn],
    pk_cols: list[str],
    dialect: str,
    table_name: str,
    inspector: Inspector,
) -> list[ColumnSchema]:
    """Convert SQLAlchemy column dicts to ``ColumnSchema`` list."""
    columns: list[ColumnSchema] = []
    for col_info in raw_columns:
        sa_type = col_info["type"]
        raw_type = _get_raw_type(sa_type)
        col_type, meta = normalize_type(sa_type, dialect, raw_type)

        name: str = col_info["name"]
        is_pk = name in pk_cols
        autoincrement = _detect_autoincrement(col_info, is_pk, pk_cols, dialect)

        default_val = _extract_default(col_info.get("default"))

        # CHECK constraint → enum_values
        check_constraint, enum_values = _extract_check_enum(table_name, name, inspector)

        # Merge enum_values from type normalization
        if "enum_values" in meta:
            enum_values = meta["enum_values"]
            col_type = ColumnType.ENUM

        columns.append(
            ColumnSchema(
                name=name,
                data_type=col_type,
                raw_type=raw_type,
                nullable=col_info.get("nullable", True),
                primary_key=is_pk,
                autoincrement=autoincrement,
                default=default_val,
                max_length=meta.get("max_length"),
                precision=meta.get("precision"),
                scale=meta.get("scale"),
                enum_values=enum_values,
                check_constraint=check_constraint,
            )
        )
    return columns


def _get_raw_type(sa_type: Any) -> str:
    """Extract the raw DDL type string from a SQLAlchemy type."""
    try:
        compiled = sa_type.compile()
        return str(compiled)
    except (TypeError, AttributeError, sa.exc.CompileError):
        return str(sa_type)


def _detect_autoincrement(
    col_info: ReflectedColumn,
    is_pk: bool,
    pk_cols: list[str],
    dialect: str,
) -> bool:
    """Detect autoincrement — SQLite ``INTEGER PRIMARY KEY`` is rowid alias."""
    if col_info.get("autoincrement", False) is True:
        return True
    if dialect == "sqlite" and is_pk and len(pk_cols) == 1:
        sa_type = col_info["type"]
        if isinstance(sa_type, sa.types.Integer) and not isinstance(
            sa_type, (sa.types.BigInteger, sa.types.SmallInteger)
        ):
            return True
    return False


def _extract_default(raw_default: Any) -> str | None:
    """Normalize a column default to a string or None."""
    if raw_default is None:
        return None
    return str(raw_default)


def _extract_check_enum(
    table_name: str,
    column_name: str,
    inspector: Inspector,
) -> tuple[str | None, list[str] | None]:
    """Extract CHECK constraints and detect ``col IN ('a','b','c')`` patterns."""
    try:
        check_constraints = inspector.get_check_constraints(table_name)
    except NotImplementedError:
        return None, None

    for cc in check_constraints:
        sqltext: str = cc.get("sqltext", "")
        # Match patterns like: status IN ('draft', 'active', 'archived')
        pattern = rf"{re.escape(column_name)}\s+IN\s*\(([^)]+)\)"
        match = re.search(pattern, sqltext, re.IGNORECASE)
        if match:
            raw_values = match.group(1)
            values = re.findall(r"'([^']*)'", raw_values)
            if values:
                return sqltext, sorted(values)
            return sqltext, None
    return None, None


def _collect_unique_columns(
    raw_uniques: list[ReflectedUniqueConstraint],
    raw_indexes: list[ReflectedIndex],
    table_name: str,
    inspector: Inspector,
) -> set[str]:
    """Collect column names that have single-column unique constraints.

    Checks explicit unique constraints, unique indexes, and (for SQLite)
    autoindexes created by inline ``UNIQUE`` column definitions — which
    SQLAlchemy's ``get_indexes()`` intentionally filters out.
    """
    unique_cols: set[str] = set()
    # Explicit unique constraints
    for uc in raw_uniques:
        cols: list[str] = uc.get("column_names", [])
        if len(cols) == 1:
            unique_cols.add(cols[0])
    # Unique indexes (user-created)
    for idx in raw_indexes:
        if idx.get("unique", False):
            idx_cols = [c for c in idx.get("column_names", []) if c is not None]
            if len(idx_cols) == 1:
                unique_cols.add(idx_cols[0])
    # SQLite autoindexes for inline UNIQUE (not returned by get_indexes)
    if inspector.dialect.name == "sqlite":
        unique_cols |= _sqlite_autoindex_unique_columns(table_name, inspector)
    return unique_cols


def _sqlite_autoindex_unique_columns(
    table_name: str,
    inspector: Inspector,
) -> set[str]:
    """Query ``PRAGMA index_list`` to find columns with SQLite autoindexes.

    Both ``table_name`` (from ``inspector.get_table_names()``) and
    ``idx_name`` (from ``PRAGMA index_list`` results) originate from
    SQLite's internal ``sqlite_master``.  We validate identifiers
    as defense-in-depth before interpolation.
    """
    unique_cols: set[str] = set()
    bind = inspector.bind
    if bind is None or not isinstance(bind, sa.engine.Engine):
        return unique_cols
    if not _is_safe_identifier(table_name):
        return unique_cols
    with bind.connect() as conn:
        rows = conn.execute(sa.text(f'PRAGMA index_list("{table_name}")')).fetchall()
        for row in rows:
            idx_name: str = row[1]
            is_unique: int = row[2]
            if (
                is_unique
                and idx_name.startswith("sqlite_autoindex_")
                and _is_safe_identifier(idx_name)
            ):
                info = conn.execute(sa.text(f'PRAGMA index_info("{idx_name}")')).fetchall()
                if len(info) == 1:
                    unique_cols.add(info[0][2])
    return unique_cols


_SAFE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_. ]*$")


def _is_safe_identifier(name: str) -> bool:
    """Allow only identifiers matching a safe allowlist pattern."""
    return bool(_SAFE_IDENT_RE.match(name)) and len(name) <= 128


def _apply_unique_flags(
    columns: list[ColumnSchema],
    unique_cols: set[str],
) -> list[ColumnSchema]:
    """Return a new column list with unique flags applied."""
    if not unique_cols:
        return columns
    return [
        col.model_copy(update={"unique": True}) if col.name in unique_cols else col
        for col in columns
    ]


def _build_foreign_keys(raw_fks: list[ReflectedForeignKeyConstraint]) -> list[ForeignKeySchema]:
    """Convert SQLAlchemy FK dicts to ``ForeignKeySchema`` list."""
    return [
        ForeignKeySchema(
            name=fk.get("name"),
            columns=fk["constrained_columns"],
            ref_table=fk["referred_table"],
            ref_columns=fk["referred_columns"],
            on_delete=fk.get("options", {}).get("ondelete"),
            on_update=fk.get("options", {}).get("onupdate"),
        )
        for fk in raw_fks
    ]


def _build_indexes(raw_indexes: list[ReflectedIndex]) -> list[IndexSchema]:
    """Convert SQLAlchemy index dicts to ``IndexSchema`` list."""
    return [
        IndexSchema(
            name=idx.get("name"),
            columns=[c for c in idx.get("column_names", []) if c is not None],
            unique=idx.get("unique", False),
        )
        for idx in raw_indexes
    ]
