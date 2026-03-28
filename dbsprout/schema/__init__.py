"""Schema input stage — unified database schema models and introspection."""

from __future__ import annotations

from dbsprout.schema.graph import FKGraph
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    IndexSchema,
    TableSchema,
)

__all__ = [
    "ColumnSchema",
    "ColumnType",
    "DatabaseSchema",
    "FKGraph",
    "ForeignKeySchema",
    "IndexSchema",
    "TableSchema",
]

try:
    from dbsprout.schema.dialect import normalize_type
    from dbsprout.schema.introspect import introspect

    __all__ += ["introspect", "normalize_type"]
except ImportError:  # pragma: no cover — sqlalchemy not installed
    pass
