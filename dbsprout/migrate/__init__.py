"""Migration awareness module — schema snapshots, diff, incremental seeding."""

from dbsprout.migrate.differ import ENUM_TABLE_SENTINEL, SchemaDiffer
from dbsprout.migrate.models import SchemaChange, SchemaChangeType
from dbsprout.migrate.update_models import (
    PlannedAction,
    UpdateAction,
    UpdatePlan,
    UpdateResult,
)
from dbsprout.migrate.updater import IncrementalUpdater

__all__ = [
    "ENUM_TABLE_SENTINEL",
    "IncrementalUpdater",
    "PlannedAction",
    "SchemaChange",
    "SchemaChangeType",
    "SchemaDiffer",
    "UpdateAction",
    "UpdatePlan",
    "UpdateResult",
]
