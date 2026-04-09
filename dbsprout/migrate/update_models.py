"""Update models for incremental seed data updates.

Defines the action types, planning, and result models used by
``IncrementalUpdater`` to classify and execute schema change actions.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from dbsprout.migrate.models import SchemaChange  # noqa: TC001 — Pydantic needs at runtime


class UpdateAction(str, Enum):
    """Classification of the update action for a schema change."""

    GENERATE_COLUMN = "generate_column"
    DROP_COLUMN = "drop_column"
    GENERATE_TABLE = "generate_table"
    DROP_TABLE = "drop_table"
    REGENERATE_COLUMN = "regenerate_column"
    VALIDATE_FK = "validate_fk"
    DEDUPLICATE = "deduplicate"
    REPLACE_ENUM = "replace_enum"
    NO_ACTION = "no_action"


class PlannedAction(BaseModel):
    """A single planned update action for a schema change."""

    model_config = ConfigDict(frozen=True)

    change: SchemaChange
    action: UpdateAction
    description: str


class UpdatePlan(BaseModel):
    """A plan of actions to apply to existing seed data."""

    model_config = ConfigDict(frozen=True)

    actions: list[PlannedAction] = []


class UpdateResult(BaseModel):
    """Result of applying an update plan to existing seed data."""

    model_config = ConfigDict(frozen=True)

    tables_data: dict[str, list[dict[str, Any]]]
    actions_applied: list[PlannedAction]
    rows_modified: int = 0
    rows_added: int = 0
    rows_removed: int = 0
    tables_added: list[str] = []
    tables_removed: list[str] = []
