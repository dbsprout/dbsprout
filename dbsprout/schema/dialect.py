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
    (sa_types.Text, ColumnType.TEXT),
]


def normalize_type(
    sa_type: sa_types.TypeEngine[Any],
    dialect: str,  # noqa: ARG001
    raw_type: str,
) -> tuple[ColumnType, dict[str, Any]]:
    """Map a SQLAlchemy type + raw DDL string to ``(ColumnType, metadata)``.

    Parameters
    ----------
    sa_type:
        The SQLAlchemy ``TypeEngine`` instance from column reflection.
    dialect:
        The database dialect name (e.g. ``"sqlite"``, ``"postgresql"``).
        Reserved for future dialect-specific overrides.
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

    # Simple 1:1 type mappings (no metadata extraction needed)
    for sa_cls, col_type in _SIMPLE_TYPE_MAP:
        if isinstance(sa_type, sa_cls):
            return col_type, {}

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


def _extract_numeric_meta(sa_type: sa_types.Numeric[Any]) -> dict[str, Any]:
    """Extract precision and scale from a Numeric/DECIMAL type."""
    meta: dict[str, Any] = {}
    if sa_type.precision is not None:
        meta["precision"] = sa_type.precision
    if sa_type.scale is not None:
        meta["scale"] = sa_type.scale
    return meta
