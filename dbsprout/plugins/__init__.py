"""Plugin discovery, registry, and Protocols for DBSprout extensions."""

from __future__ import annotations

from dbsprout.plugins.errors import PluginError, PluginValidationError
from dbsprout.plugins.protocols import (
    GenerationEngine,
    MigrationParser,
    OutputWriter,
    SchemaParser,
    SpecProvider,
)
from dbsprout.plugins.registry import PluginInfo, PluginRegistry, get_registry

__all__ = [
    "GenerationEngine",
    "MigrationParser",
    "OutputWriter",
    "PluginError",
    "PluginInfo",
    "PluginRegistry",
    "PluginValidationError",
    "SchemaParser",
    "SpecProvider",
    "get_registry",
]
