"""Unified database schema models.

All parsers and introspection backends produce these immutable Pydantic v2
models. Consumers never know the source format — the unified model is the
contract between pipeline stages.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _validate_identifier(v: str) -> str:
    """Validate a SQL identifier name."""
    if _CONTROL_CHAR_RE.search(v):
        raise ValueError("Identifier must not contain control characters")
    v = v.strip()
    if not v:
        raise ValueError("Identifier must not be empty")
    if len(v) > 128:
        raise ValueError(f"Identifier must be ≤128 chars, got {len(v)}")
    if "/" in v or "\\" in v or v.startswith("..") or v == ".":
        raise ValueError(f"Identifier {v!r} contains path-separator or path-traversal characters")
    return v


Identifier = Annotated[str, AfterValidator(_validate_identifier)]

IdentifierList = list[Identifier]

_VALID_REFERENTIAL_ACTIONS = frozenset(
    {"CASCADE", "SET NULL", "SET DEFAULT", "RESTRICT", "NO ACTION"}
)


def _normalize_referential_action(v: str | None) -> str | None:
    """Normalize and validate a SQL referential action."""
    if v is None:
        return None
    v = v.upper()
    if v not in _VALID_REFERENTIAL_ACTIONS:
        raise ValueError(
            f"Invalid referential action {v!r}; must be one of {sorted(_VALID_REFERENTIAL_ACTIONS)}"
        )
    return v


ReferentialAction = Annotated[str | None, BeforeValidator(_normalize_referential_action)]

_VALID_DEFER_TIMINGS = frozenset({"DEFERRED", "IMMEDIATE"})


def _normalize_defer_timing(v: str | None) -> str | None:
    """Normalize and validate a SQL FK ``INITIALLY`` timing."""
    if v is None:
        return None
    v = v.upper()
    if v not in _VALID_DEFER_TIMINGS:
        raise ValueError(
            f"Invalid INITIALLY timing {v!r}; must be one of {sorted(_VALID_DEFER_TIMINGS)}"
        )
    return v


DeferTiming = Annotated[str | None, BeforeValidator(_normalize_defer_timing)]


def _quote_ident(name: str) -> str:
    """Double-quote a SQL identifier, escaping internal double quotes."""
    return '"' + name.replace('"', '""') + '"'


class ColumnType(str, Enum):
    """Normalized column types across all database dialects."""

    INTEGER = "integer"
    BIGINT = "bigint"
    SMALLINT = "smallint"
    FLOAT = "float"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    VARCHAR = "varchar"
    TEXT = "text"
    DATE = "date"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"
    TIME = "time"
    UUID = "uuid"
    JSON = "json"
    BINARY = "binary"
    ENUM = "enum"
    ARRAY = "array"
    UNKNOWN = "unknown"


class ColumnSchema(BaseModel):
    """A single column in a database table."""

    model_config = ConfigDict(frozen=True)

    name: Identifier
    data_type: ColumnType
    raw_type: str = ""
    nullable: bool = True
    primary_key: bool = False
    unique: bool = False
    autoincrement: bool = False
    default: str | None = None
    max_length: int | None = None
    precision: int | None = None
    scale: int | None = None
    enum_values: list[str] | None = None
    check_constraint: str | None = None
    comment: str | None = None


class ForeignKeySchema(BaseModel):
    """A foreign key constraint between tables."""

    model_config = ConfigDict(frozen=True)

    name: str | None = None
    columns: IdentifierList
    ref_table: Identifier
    ref_columns: IdentifierList
    on_delete: ReferentialAction = None
    on_update: ReferentialAction = None
    deferrable: bool = False
    initially: DeferTiming = None


class IndexSchema(BaseModel):
    """A database index."""

    model_config = ConfigDict(frozen=True)

    name: str | None = None
    columns: IdentifierList
    unique: bool = False


class TableSchema(BaseModel):
    """A database table with columns, keys, and indexes."""

    model_config = ConfigDict(frozen=True)

    name: Identifier
    columns: list[ColumnSchema]
    primary_key: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKeySchema] = Field(default_factory=list)
    indexes: list[IndexSchema] = Field(default_factory=list)
    comment: str | None = None
    row_count_hint: int | None = None

    def get_column(self, name: str) -> ColumnSchema | None:
        """Find a column by name, or ``None`` if not found."""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    @property
    def fk_parent_tables(self) -> list[str]:
        """Unique referenced tables, excluding self-references, sorted."""
        return sorted({fk.ref_table for fk in self.foreign_keys if fk.ref_table != self.name})

    @property
    def is_junction_table(self) -> bool:
        """True when all PK columns are FK columns and there are ≥2 FKs."""
        if len(self.primary_key) == 0 or len(self.foreign_keys) < 2:
            return False
        fk_cols = {col for fk in self.foreign_keys for col in fk.columns}
        pk_cols = set(self.primary_key)
        return pk_cols.issubset(fk_cols)


class DatabaseSchema(BaseModel):
    """Complete database schema — the unified output of all parsers."""

    model_config = ConfigDict(frozen=True)

    tables: list[TableSchema]
    enums: dict[str, list[str]] = Field(default_factory=dict)
    dialect: str | None = None
    source: str | None = None
    source_file: str | None = None

    def get_table(self, name: str) -> TableSchema | None:
        """Find a table by name, or ``None`` if not found."""
        for table in self.tables:
            if table.name == name:
                return table
        return None

    def table_names(self) -> list[str]:
        """Return table names in definition order."""
        return [t.name for t in self.tables]

    def schema_hash(self) -> str:
        """Deterministic SHA-256[:16] hex digest of schema content.

        Ordering-independent: tables and columns are sorted by name.
        Metadata fields (dialect, source, comments, raw_type) are excluded
        so the same logical schema always produces the same hash.
        """
        canonical = self._canonical_dict()
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_ddl(self, dialect: str | None = None) -> str:  # noqa: ARG002
        """Generate SQL DDL for LLM prompt construction.

        Produces generic CREATE TABLE statements. Not intended to be
        directly executable — just structurally correct for LLM context.
        """
        parts: list[str] = []
        for table in sorted(self.tables, key=lambda t: t.name):
            parts.append(self._table_ddl(table))
        return "\n\n".join(parts)

    # ── Private helpers ──────────────────────────────────────────────

    def _canonical_dict(self) -> dict[str, Any]:
        """Order-independent canonical representation for hashing."""
        tables = sorted(
            [self._canonical_table(t) for t in self.tables],
            key=lambda t: str(t["name"]),
        )
        enums = {k: sorted(v) for k, v in sorted(self.enums.items())}
        return {"tables": tables, "enums": enums}

    @staticmethod
    def _canonical_table(t: TableSchema) -> dict[str, Any]:
        columns = sorted(
            [DatabaseSchema._canonical_column(c) for c in t.columns],
            key=lambda c: str(c["name"]),
        )
        fks = sorted(
            [
                {
                    "columns": sorted(fk.columns),
                    "ref_table": fk.ref_table,
                    "ref_columns": sorted(fk.ref_columns),
                    "on_delete": fk.on_delete,
                    "on_update": fk.on_update,
                    "deferrable": fk.deferrable,
                    "initially": fk.initially,
                }
                for fk in t.foreign_keys
            ],
            key=lambda f: (f["ref_table"], f["columns"]),
        )
        indexes = sorted(
            [{"columns": sorted(idx.columns), "unique": idx.unique} for idx in t.indexes],
            key=lambda i: (i["columns"], i["unique"]),
        )
        return {
            "name": t.name,
            "columns": columns,
            "primary_key": sorted(t.primary_key),
            "foreign_keys": fks,
            "indexes": indexes,
        }

    @staticmethod
    def _canonical_column(c: ColumnSchema) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": c.name,
            "data_type": c.data_type.value,
            "nullable": c.nullable,
            "primary_key": c.primary_key,
            "unique": c.unique,
            "autoincrement": c.autoincrement,
        }
        if c.default is not None:
            d["default"] = c.default
        if c.max_length is not None:
            d["max_length"] = c.max_length
        if c.precision is not None:
            d["precision"] = c.precision
        if c.scale is not None:
            d["scale"] = c.scale
        if c.enum_values is not None:
            d["enum_values"] = sorted(c.enum_values)
        if c.check_constraint is not None:
            d["check_constraint"] = c.check_constraint
        return d

    @staticmethod
    def _table_ddl(table: TableSchema) -> str:
        lines: list[str] = []
        single_pk = len(table.primary_key) == 1

        for col in table.columns:
            parts: list[str] = [f"    {_quote_ident(col.name)}"]
            parts.append(DatabaseSchema._col_type_ddl(col))
            if not col.nullable:
                parts.append("NOT NULL")
            if col.primary_key and single_pk:
                parts.append("PRIMARY KEY")
            if col.autoincrement:
                parts.append("AUTOINCREMENT")
            if col.unique:
                parts.append("UNIQUE")
            if col.default is not None:
                parts.append(f"DEFAULT {col.default}")
            if col.check_constraint is not None:
                parts.append(f"CHECK ({col.check_constraint})")
            lines.append(" ".join(parts))

        if len(table.primary_key) > 1:
            pk_cols = ", ".join(_quote_ident(c) for c in table.primary_key)
            lines.append(f"    PRIMARY KEY ({pk_cols})")

        for fk in table.foreign_keys:
            fk_cols = ", ".join(_quote_ident(c) for c in fk.columns)
            ref_cols = ", ".join(_quote_ident(c) for c in fk.ref_columns)
            ref_table = _quote_ident(fk.ref_table)
            fk_line = f"    FOREIGN KEY ({fk_cols}) REFERENCES {ref_table} ({ref_cols})"
            if fk.on_delete is not None:
                fk_line += f" ON DELETE {fk.on_delete}"
            if fk.on_update is not None:
                fk_line += f" ON UPDATE {fk.on_update}"
            lines.append(fk_line)

        body = ",\n".join(lines)
        return f"CREATE TABLE {_quote_ident(table.name)} (\n{body}\n);"

    @staticmethod
    def _col_type_ddl(col: ColumnSchema) -> str:
        t = col.data_type.value.upper()
        if col.data_type == ColumnType.VARCHAR and col.max_length is not None:
            return f"VARCHAR({col.max_length})"
        if col.data_type == ColumnType.DECIMAL and col.precision is not None:
            if col.scale is not None:
                return f"DECIMAL({col.precision},{col.scale})"
            return f"DECIMAL({col.precision})"
        return t
