"""Schema change models for migration-aware diffing.

Defines the ``SchemaChangeType`` enum and ``SchemaChange`` model used by
``SchemaDiffer`` to represent structural differences between two
``DatabaseSchema`` snapshots.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class SchemaChangeType(str, Enum):
    """Classification of a structural schema change."""

    TABLE_ADDED = "table_added"
    TABLE_REMOVED = "table_removed"
    COLUMN_ADDED = "column_added"
    COLUMN_REMOVED = "column_removed"
    COLUMN_TYPE_CHANGED = "column_type_changed"
    COLUMN_NULLABILITY_CHANGED = "column_nullability_changed"
    COLUMN_DEFAULT_CHANGED = "column_default_changed"
    FOREIGN_KEY_ADDED = "foreign_key_added"
    FOREIGN_KEY_REMOVED = "foreign_key_removed"
    INDEX_ADDED = "index_added"
    INDEX_REMOVED = "index_removed"
    ENUM_CHANGED = "enum_changed"


class SchemaChange(BaseModel):
    """A single structural change between two database schemas.

    Carries enough context in ``detail`` for downstream consumers
    (e.g. ``IncrementalUpdater``) to determine the correct update action
    without re-inspecting the original schemas.
    """

    model_config = ConfigDict(frozen=True)

    change_type: SchemaChangeType
    table_name: str
    column_name: str | None = None
    old_value: str | None = None
    new_value: str | None = None
    detail: dict[str, Any] | None = None
