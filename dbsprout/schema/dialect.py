"""Type normalization across database dialects.

Maps SQLAlchemy type objects to the unified ``ColumnType`` enum.
Dialect-specific helpers live here; the public entry point is
``normalize_type()``.
"""

from __future__ import annotations

from typing import Any

import sqlalchemy.types as sa_types

from dbsprout.schema.models import ColumnType

# Ordered dispatch: (SA type class, ColumnType).
# Order matters — subclasses must appear before parents.
# Float before Numeric so that Float doesn't fall into DECIMAL.
# Text before String so that Text doesn't fall into VARCHAR.
_SIMPLE_TYPE_MAP: list[tuple[type[sa_types.TypeEngine[Any]], ColumnType]] = [
    (sa_types.Boolean, ColumnType.BOOLEAN),
    (sa_types.BigInteger, ColumnType.BIGINT),
    (sa_types.SmallInteger, ColumnType.SMALLINT),
    (sa_types.Integer, ColumnType.INTEGER),
    (sa_types.Float, ColumnType.FLOAT),
    (sa_types.TIMESTAMP, ColumnType.TIMESTAMP),
    (sa_types.DateTime, ColumnType.DATETIME),
    (sa_types.Date, ColumnType.DATE),
    (sa_types.Time, ColumnType.TIME),
    (sa_types.LargeBinary, ColumnType.BINARY),
    (sa_types.JSON, ColumnType.JSON),
    (sa_types.ARRAY, ColumnType.ARRAY),
    (sa_types.Uuid, ColumnType.UUID),
    (sa_types.Text, ColumnType.TEXT),
]

# PG types that do NOT inherit from any generic SA type and need explicit
# mapping. These are only checked when dialect == "postgresql".
_PG_TYPE_MAP: dict[str, ColumnType] = {
    "INET": ColumnType.VARCHAR,
    "CIDR": ColumnType.VARCHAR,
    "MACADDR": ColumnType.VARCHAR,
    "MACADDR8": ColumnType.VARCHAR,
    "MONEY": ColumnType.DECIMAL,
    "INTERVAL": ColumnType.VARCHAR,
    "TSVECTOR": ColumnType.TEXT,
    "TSQUERY": ColumnType.TEXT,
    "HSTORE": ColumnType.JSON,
}

# MySQL types that do NOT inherit from the expected generic SA type.
_MYSQL_TYPE_MAP: dict[str, ColumnType] = {
    "MEDIUMTEXT": ColumnType.TEXT,
    "LONGTEXT": ColumnType.TEXT,
    "TINYTEXT": ColumnType.TEXT,
    "TINYBLOB": ColumnType.BINARY,
    "MEDIUMBLOB": ColumnType.BINARY,
    "LONGBLOB": ColumnType.BINARY,
    "YEAR": ColumnType.INTEGER,
    "BIT": ColumnType.INTEGER,
}

# MSSQL types that either have no generic SA superclass or need an override.
# UNIQUEIDENTIFIER, BIT, IMAGE, NVARCHAR, NCHAR, NTEXT, XML, SMALLDATETIME
# are handled by generic SA dispatch (Uuid, Boolean, LargeBinary, String,
# Text, DateTime) and do NOT need entries here.
_MSSQL_TYPE_MAP: dict[str, ColumnType] = {
    "DATETIME2": ColumnType.TIMESTAMP,
    "DATETIMEOFFSET": ColumnType.TIMESTAMP,
    "MONEY": ColumnType.DECIMAL,
    "SMALLMONEY": ColumnType.DECIMAL,
}

# Registry of dialect-specific type maps.
_DIALECT_TYPE_MAPS: dict[str, dict[str, ColumnType]] = {
    "postgresql": _PG_TYPE_MAP,
    "mysql": _MYSQL_TYPE_MAP,
    "mssql": _MSSQL_TYPE_MAP,
}


def normalize_type(
    sa_type: sa_types.TypeEngine[Any],
    dialect: str,
    raw_type: str,
) -> tuple[ColumnType, dict[str, Any]]:
    """Map a SQLAlchemy type + raw DDL string to ``(ColumnType, metadata)``.

    Parameters
    ----------
    sa_type:
        The SQLAlchemy ``TypeEngine`` instance from column reflection.
    dialect:
        The database dialect name (e.g. ``"sqlite"``, ``"postgresql"``).
        Used to gate dialect-specific type mappings.
    raw_type:
        The original type string as declared in DDL (preserved verbatim).

    Returns
    -------
    tuple[ColumnType, dict[str, Any]]
        The normalized column type and a metadata dict that may contain
        ``max_length``, ``precision``, ``scale``, or ``enum_values``.
    """
    # Raw-type overrides (checked before SA dispatch)
    if raw_type.upper() == "UUID":
        return ColumnType.UUID, {}

    # Enum
    if isinstance(sa_type, sa_types.Enum):
        return ColumnType.ENUM, {"enum_values": sorted(sa_type.enums)}

    # Dialect-specific special cases + simple 1:1 type mappings
    dispatch = _dispatch_type(sa_type, dialect)
    if dispatch is not None:
        return dispatch

    # Numeric/DECIMAL family (Float already handled above in dispatch table)
    if isinstance(sa_type, (sa_types.DECIMAL, sa_types.Numeric)):
        return ColumnType.DECIMAL, _extract_numeric_meta(sa_type)

    # String/VARCHAR family (Text already handled above in dispatch table)
    if isinstance(sa_type, (sa_types.String, sa_types.VARCHAR)):
        meta: dict[str, Any] = {}
        if sa_type.length is not None:
            meta["max_length"] = sa_type.length
        return ColumnType.VARCHAR, meta

    # Fallback
    return ColumnType.UNKNOWN, {}


def _dispatch_type(
    sa_type: sa_types.TypeEngine[Any],
    dialect: str,
) -> tuple[ColumnType, dict[str, Any]] | None:
    """Try dialect-specific special cases, then simple type map dispatch."""
    special = _normalize_dialect_special(sa_type, dialect)
    if special is not None:
        return special
    simple = _match_simple_type(sa_type, dialect)
    if simple is not None:
        return simple, {}
    return None


def _normalize_dialect_special(
    sa_type: sa_types.TypeEngine[Any],
    dialect: str,
) -> tuple[ColumnType, dict[str, Any]] | None:
    """Handle dialect-specific types that need metadata extraction.

    Returns None if no match — falls through to generic dispatch.
    """
    if dialect != "mysql":
        return None
    type_name = type(sa_type).__name__
    # MySQL TINYINT: display_width=1 → BOOLEAN, else → SMALLINT
    if type_name == "TINYINT":
        if getattr(sa_type, "display_width", None) == 1:
            return ColumnType.BOOLEAN, {}
        return ColumnType.SMALLINT, {}
    # MySQL SET: multi-value selection, report as ENUM with allowed values
    if type_name == "SET":
        values = getattr(sa_type, "values", None)
        meta: dict[str, Any] = {"enum_values": sorted(values)} if values else {}
        return ColumnType.ENUM, meta
    return None


def _match_simple_type(
    sa_type: sa_types.TypeEngine[Any],
    dialect: str,
) -> ColumnType | None:
    """Check dialect-specific types, then generic SA type dispatch. Returns None if no match."""
    dialect_map = _DIALECT_TYPE_MAPS.get(dialect)
    if dialect_map is not None:
        match = dialect_map.get(type(sa_type).__name__)
        if match is not None:
            return match
    for sa_cls, col_type in _SIMPLE_TYPE_MAP:
        if isinstance(sa_type, sa_cls):
            return col_type
    return None


def _extract_numeric_meta(sa_type: sa_types.Numeric[Any]) -> dict[str, Any]:
    """Extract precision and scale from a Numeric/DECIMAL type."""
    meta: dict[str, Any] = {}
    if sa_type.precision is not None:
        meta["precision"] = sa_type.precision
    if sa_type.scale is not None:
        meta["scale"] = sa_type.scale
    return meta
