"""SQL DDL file parser — produces a ``DatabaseSchema`` from DDL text.

Uses sqlglot to parse CREATE TABLE, ALTER TABLE, and CREATE INDEX
statements across SQLite, PostgreSQL, and MySQL dialects.
"""

from __future__ import annotations

import re
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.expressions import DataType as DType

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    IndexSchema,
    TableSchema,
)

# ── Type mapping ─────────────────────────────────────────────────────────

_DTYPE_MAP: dict[Any, ColumnType] = {  # keys are DType.Type enum members
    DType.Type.INT: ColumnType.INTEGER,
    DType.Type.BIGINT: ColumnType.BIGINT,
    DType.Type.SMALLINT: ColumnType.SMALLINT,
    DType.Type.TINYINT: ColumnType.SMALLINT,
    DType.Type.FLOAT: ColumnType.FLOAT,
    DType.Type.DOUBLE: ColumnType.FLOAT,
    DType.Type.DECIMAL: ColumnType.DECIMAL,
    DType.Type.BOOLEAN: ColumnType.BOOLEAN,
    DType.Type.BIT: ColumnType.INTEGER,
    DType.Type.VARCHAR: ColumnType.VARCHAR,
    DType.Type.NVARCHAR: ColumnType.VARCHAR,
    DType.Type.CHAR: ColumnType.VARCHAR,
    DType.Type.NCHAR: ColumnType.VARCHAR,
    DType.Type.TEXT: ColumnType.TEXT,
    DType.Type.MEDIUMTEXT: ColumnType.TEXT,
    DType.Type.LONGTEXT: ColumnType.TEXT,
    DType.Type.DATE: ColumnType.DATE,
    DType.Type.DATETIME: ColumnType.DATETIME,
    DType.Type.TIMESTAMP: ColumnType.TIMESTAMP,
    DType.Type.TIMESTAMPTZ: ColumnType.TIMESTAMP,
    DType.Type.TIME: ColumnType.TIME,
    DType.Type.UUID: ColumnType.UUID,
    DType.Type.JSON: ColumnType.JSON,
    DType.Type.JSONB: ColumnType.JSON,
    DType.Type.BINARY: ColumnType.BINARY,
    DType.Type.VARBINARY: ColumnType.BINARY,
    DType.Type.IMAGE: ColumnType.BINARY,
    DType.Type.ENUM: ColumnType.ENUM,
    DType.Type.ARRAY: ColumnType.ARRAY,
    DType.Type.SERIAL: ColumnType.INTEGER,
    DType.Type.BIGSERIAL: ColumnType.BIGINT,
    DType.Type.SMALLSERIAL: ColumnType.SMALLINT,
    DType.Type.MONEY: ColumnType.DECIMAL,
    DType.Type.INET: ColumnType.VARCHAR,
    DType.Type.HSTORE: ColumnType.JSON,
    DType.Type.INTERVAL: ColumnType.VARCHAR,
}

_SERIAL_TYPES = frozenset({DType.Type.SERIAL, DType.Type.BIGSERIAL, DType.Type.SMALLSERIAL})

# Dialect normalization: sqlglot dialect names → our dialect names
_DIALECT_NORMALIZE: dict[str, str] = {
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "mysql": "mysql",
    "sqlite": "sqlite",
}


# ── Public API ───────────────────────────────────────────────────────────


def parse_ddl(
    sql_text: str,
    *,
    dialect: str | None = None,
    source_file: str | None = None,
) -> DatabaseSchema:
    """Parse SQL DDL text into a ``DatabaseSchema``.

    Parameters
    ----------
    sql_text:
        Raw SQL text containing CREATE TABLE, ALTER TABLE, CREATE INDEX.
    dialect:
        SQL dialect (``"postgres"``, ``"mysql"``, ``"sqlite"``).
        If ``None``, auto-detected from DDL content.
    source_file:
        Original file path for metadata.

    Raises
    ------
    ValueError
        If no CREATE TABLE statements found.
    """
    if dialect is None:
        dialect = _detect_dialect(sql_text)

    sqlglot_dialect = dialect if dialect else None
    statements = sqlglot.parse(sql_text, dialect=sqlglot_dialect)

    tables: dict[str, TableSchema] = {}
    indexes: dict[str, list[IndexSchema]] = {}

    # First pass: CREATE TABLE
    for stmt in statements:
        if stmt is None:
            continue
        if isinstance(stmt, exp.Create) and isinstance(stmt.this, exp.Schema):
            table = _extract_table(stmt)
            if table is not None:
                tables[table.name] = table
        elif isinstance(stmt, exp.Create) and isinstance(stmt.this, exp.Index):
            _extract_index(stmt, indexes)

    # Second pass: ALTER TABLE ADD FK
    for stmt in statements:
        if stmt is None:
            continue
        if isinstance(stmt, exp.Alter):
            _merge_alter_fks(stmt, tables)

    if not tables:
        msg = "No CREATE TABLE statements found in DDL"
        raise ValueError(msg)

    # Merge indexes into tables
    for table_name, idx_list in indexes.items():
        if table_name in tables:
            existing = tables[table_name]
            merged = [*list(existing.indexes), *idx_list]
            tables[table_name] = existing.model_copy(update={"indexes": merged})

    normalized_dialect = _DIALECT_NORMALIZE.get(dialect, dialect) if dialect else None

    return DatabaseSchema(
        tables=list(tables.values()),
        dialect=normalized_dialect,
        source="ddl",
        source_file=source_file,
    )


# ── Dialect detection ────────────────────────────────────────────────────


def _detect_dialect(sql_text: str) -> str | None:
    """Auto-detect SQL dialect from DDL content."""
    upper = sql_text.upper()
    if re.search(r"\bSERIAL\b", upper) or re.search(r"\bBIGSERIAL\b", upper):
        return "postgres"
    if re.search(r"\bAUTO_INCREMENT\b", upper):
        return "mysql"
    if "`" in sql_text:
        return "mysql"
    if re.search(r"\bAUTOINCREMENT\b", upper):
        return "sqlite"
    return None


# ── Table extraction ─────────────────────────────────────────────────────


def _extract_table(create_stmt: exp.Create) -> TableSchema | None:
    """Extract a TableSchema from a CREATE TABLE statement."""
    schema_expr = create_stmt.this
    if not isinstance(schema_expr, exp.Schema):  # pragma: no cover
        return None

    table_expr = schema_expr.this
    if not isinstance(table_expr, exp.Table):  # pragma: no cover
        return None

    table_name = table_expr.name

    columns: list[ColumnSchema] = []
    pk_cols: list[str] = []
    foreign_keys: list[ForeignKeySchema] = []

    for expr in schema_expr.expressions:
        if isinstance(expr, exp.ColumnDef):
            col, is_pk, fk = _extract_column(expr)
            columns.append(col)
            if is_pk:
                pk_cols.append(col.name)
            if fk is not None:
                foreign_keys.append(fk)
        elif isinstance(expr, exp.PrimaryKey):
            for ident in expr.expressions:
                name = ident.name if isinstance(ident, exp.Identifier) else str(ident)
                if name not in pk_cols:
                    pk_cols.append(name)
        elif isinstance(expr, exp.ForeignKey):
            fk = _extract_fk_from_expr(expr)
            if fk is not None:
                foreign_keys.append(fk)
        elif isinstance(expr, exp.UniqueColumnConstraint):
            _mark_unique_from_constraint(expr, columns)
        elif isinstance(expr, exp.Constraint):
            # Named constraint wrapping a ForeignKey or UniqueColumnConstraint
            kind = expr.find(exp.ForeignKey)
            if kind is not None:
                fk = _extract_fk_from_expr(kind)
                if fk is not None:
                    foreign_keys.append(fk)

    return TableSchema(
        name=table_name,
        columns=columns,
        primary_key=pk_cols,
        foreign_keys=foreign_keys,
    )


# ── Column extraction ────────────────────────────────────────────────────


def _extract_column(
    col_def: exp.ColumnDef,
) -> tuple[ColumnSchema, bool, ForeignKeySchema | None]:
    """Extract column info, PK flag, and optional inline FK."""
    col_name = col_def.name
    dtype = col_def.args.get("kind")

    col_type = ColumnType.UNKNOWN
    raw_type = ""
    max_length: int | None = None
    precision: int | None = None
    scale: int | None = None
    autoincrement = False

    if isinstance(dtype, exp.DataType):
        raw_type = dtype.sql()
        mapped = _DTYPE_MAP.get(dtype.this)
        if mapped is not None:
            col_type = mapped
        if dtype.this in _SERIAL_TYPES:
            autoincrement = True
        # Extract type parameters
        params = dtype.expressions
        if params and col_type is ColumnType.VARCHAR:
            max_length = _param_int(params, 0)
        elif params and col_type is ColumnType.DECIMAL:
            precision = _param_int(params, 0)
            if len(params) > 1:
                scale = _param_int(params, 1)

    nullable = True
    unique = False
    is_pk = False
    default: str | None = None
    check_constraint: str | None = None
    enum_values: list[str] | None = None
    fk: ForeignKeySchema | None = None

    for constraint in col_def.args.get("constraints", []):
        kind = constraint.args.get("kind")
        if kind is None:  # pragma: no cover — constraints always have kind
            continue
        if isinstance(kind, exp.NotNullColumnConstraint):
            nullable = False
        elif isinstance(kind, exp.PrimaryKeyColumnConstraint):
            is_pk = True
            nullable = False
        elif isinstance(kind, exp.UniqueColumnConstraint):
            unique = True
        elif isinstance(kind, exp.AutoIncrementColumnConstraint):
            autoincrement = True
        elif isinstance(kind, exp.DefaultColumnConstraint):
            default = kind.this.sql() if kind.this else None
        elif isinstance(kind, exp.CheckColumnConstraint):
            check_text = kind.this.sql() if kind.this else ""
            check_constraint = check_text
            # Extract IN pattern enum values
            values = _extract_check_enum_values(check_text)
            if values:
                enum_values = values
                col_type = ColumnType.ENUM
        elif isinstance(kind, exp.Reference):
            fk = _extract_inline_fk(kind, col_name)

    return (
        ColumnSchema(
            name=col_name,
            data_type=col_type,
            raw_type=raw_type,
            nullable=nullable,
            primary_key=is_pk,
            unique=unique,
            autoincrement=autoincrement,
            default=default,
            max_length=max_length,
            precision=precision,
            scale=scale,
            enum_values=enum_values,
            check_constraint=check_constraint,
        ),
        is_pk,
        fk,
    )


# ── FK extraction ────────────────────────────────────────────────────────


def _extract_inline_fk(ref: exp.Reference, col_name: str) -> ForeignKeySchema | None:
    """Extract FK from inline REFERENCES constraint."""
    ref_target = ref.this
    if ref_target is None:  # pragma: no cover — sqlglot always produces a target
        return None

    if isinstance(ref_target, exp.Table):
        ref_table = ref_target.name
        ref_cols = [e.name for e in ref.expressions] if ref.expressions else []
    elif isinstance(ref_target, exp.Schema):
        tbl = ref_target.this
        ref_table = tbl.name if isinstance(tbl, exp.Table) else str(tbl)
        ref_cols = [e.name for e in ref_target.expressions if isinstance(e, exp.Identifier)]
    else:  # pragma: no cover — defensive guard
        return None

    on_delete, on_update = _extract_fk_actions(ref)

    return ForeignKeySchema(  # inline FK
        columns=[col_name],
        ref_table=ref_table,
        ref_columns=ref_cols if ref_cols else ["id"],
        on_delete=on_delete,
        on_update=on_update,
    )


def _extract_fk_from_expr(fk_expr: exp.ForeignKey) -> ForeignKeySchema | None:
    """Extract FK from table-level FOREIGN KEY constraint."""
    src_cols = [e.name for e in fk_expr.expressions if isinstance(e, (exp.Identifier, exp.Column))]

    ref = fk_expr.find(exp.Reference)
    if ref is None:  # pragma: no cover — FK always has a reference
        return None

    ref_target = ref.this
    if ref_target is None:  # pragma: no cover — defensive guard
        return None

    if isinstance(ref_target, exp.Table):
        ref_table = ref_target.name
        ref_cols = [e.name for e in ref.expressions] if ref.expressions else []
    elif isinstance(ref_target, exp.Schema):
        tbl = ref_target.this
        ref_table = tbl.name if isinstance(tbl, exp.Table) else str(tbl)
        ref_cols = [e.name for e in ref_target.expressions if isinstance(e, exp.Identifier)]
    else:  # pragma: no cover — defensive guard
        return None

    # Options may be on the ForeignKey node or the Reference node
    on_delete, on_update = _extract_fk_actions(fk_expr)
    if on_delete is None and on_update is None:
        on_delete, on_update = _extract_fk_actions(ref)

    return ForeignKeySchema(
        columns=src_cols,
        ref_table=ref_table,
        ref_columns=ref_cols if ref_cols else ["id"],
        on_delete=on_delete,
        on_update=on_update,
    )


def _extract_fk_actions(
    fk_or_ref: exp.ForeignKey | exp.Reference,
) -> tuple[str | None, str | None]:
    """Extract ON DELETE / ON UPDATE from FK options."""
    on_delete: str | None = None
    on_update: str | None = None

    # Options may be on the ForeignKey node or the Reference node
    options = fk_or_ref.args.get("options", [])
    for opt in options:
        opt_str = str(opt).upper().strip()
        if opt_str.startswith("ON DELETE"):
            on_delete = opt_str.removeprefix("ON DELETE").strip()
        elif opt_str.startswith("ON UPDATE"):
            on_update = opt_str.removeprefix("ON UPDATE").strip()
    return on_delete, on_update


# ── ALTER TABLE ──────────────────────────────────────────────────────────


def _merge_alter_fks(alter_stmt: exp.Alter, tables: dict[str, TableSchema]) -> None:
    """Merge ALTER TABLE ADD FOREIGN KEY into existing tables."""
    table_expr = alter_stmt.this
    if not isinstance(table_expr, exp.Table):  # pragma: no cover
        return
    table_name = table_expr.name
    if table_name not in tables:
        return

    for action in alter_stmt.args.get("actions", []):
        fk_node = None
        if isinstance(action, exp.AddConstraint):
            fk_node = action.find(exp.ForeignKey)
        if fk_node is None:
            continue
        fk = _extract_fk_from_expr(fk_node)
        if fk is None:  # pragma: no cover — FK always parseable if found
            continue
        existing = tables[table_name]
        tables[table_name] = existing.model_copy(
            update={"foreign_keys": [*list(existing.foreign_keys), fk]}
        )


# ── Index extraction ─────────────────────────────────────────────────────


def _extract_index(
    create_stmt: exp.Create,
    indexes: dict[str, list[IndexSchema]],
) -> None:
    """Extract index from CREATE [UNIQUE] INDEX statement."""
    idx_expr = create_stmt.this
    if not isinstance(idx_expr, exp.Index):  # pragma: no cover
        return

    idx_name = idx_expr.alias_or_name or idx_expr.name
    table_expr = idx_expr.args.get("table")
    if not isinstance(table_expr, exp.Table):  # pragma: no cover
        return
    table_name = table_expr.name

    # Columns from the index params
    params = idx_expr.args.get("params")
    col_names: list[str] = []
    if params:
        col_names = [c.name for c in params.find_all(exp.Column)]

    is_unique = bool(create_stmt.args.get("unique"))

    idx = IndexSchema(name=idx_name, columns=col_names, unique=is_unique)
    indexes.setdefault(table_name, []).append(idx)


# ── Helpers ──────────────────────────────────────────────────────────────


def _mark_unique_from_constraint(
    unique_expr: exp.UniqueColumnConstraint,
    columns: list[ColumnSchema],
) -> None:
    """Mark columns as unique from a table-level UNIQUE constraint."""
    col_names: set[str] = set()
    if unique_expr.this and isinstance(unique_expr.this, exp.Schema):
        for ident in unique_expr.this.expressions:
            if isinstance(ident, exp.Identifier):
                col_names.add(ident.name)
    for i, col in enumerate(columns):
        if col.name in col_names:
            columns[i] = col.model_copy(update={"unique": True})


def _param_int(params: list[Any], idx: int) -> int | None:
    """Extract integer from DataType parameters."""
    if idx >= len(params):
        return None
    param = params[idx]
    if isinstance(param, exp.DataTypeParam):
        param = param.this
    if isinstance(param, exp.Literal) and param.is_int:
        return int(param.this)
    try:  # pragma: no cover — fallback for non-standard param formats
        return int(str(param))
    except (ValueError, TypeError):
        return None


def _extract_check_enum_values(check_text: str) -> list[str] | None:
    """Extract enum values from CHECK (col IN ('a', 'b', 'c')) pattern."""
    match = re.search(r"IN\s*\(([^)]+)\)", check_text, re.IGNORECASE)
    if match:
        raw = match.group(1)
        values = re.findall(r"'([^']*)'", raw)
        if values:
            return sorted(values)
    return None
