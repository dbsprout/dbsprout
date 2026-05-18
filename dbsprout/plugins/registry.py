"""Plugin registry — lazy singleton populated from ``entry_points``."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Any, Literal

from dbsprout.plugins.discovery import _safe_label, discover
from dbsprout.plugins.errors import PluginValidationError
from dbsprout.plugins.protocols import (
    GenerationEngine,
    MigrationParser,
    OutputWriter,
    SchemaParser,
    SpecProvider,
    TrainExtractor,
)

logger = logging.getLogger(__name__)

GROUPS: dict[str, type] = {
    "dbsprout.parsers": SchemaParser,
    "dbsprout.generators": GenerationEngine,
    "dbsprout.outputs": OutputWriter,
    "dbsprout.llm_providers": SpecProvider,
    "dbsprout.migration_frameworks": MigrationParser,
    "dbsprout.train_extractors": TrainExtractor,
}


@dataclass(frozen=True)
class PluginInfo:
    group: str
    name: str
    module: str
    status: Literal["loaded", "error"]
    error: str | None
    obj: Any | None


def _module_path(obj: Any) -> str:
    mod: str | None = getattr(obj, "__module__", None)
    if mod:
        return mod
    cls = getattr(obj, "__class__", None)
    result: str = getattr(cls, "__module__", "<unknown>")
    return result


def _passes_protocol(obj: Any, protocol: type) -> bool:
    return isinstance(obj, protocol)


class PluginRegistry:
    """Holds discovered plugins, keyed by ``(group, name)``."""

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], PluginInfo] = {}
        self._build()

    def _build(self) -> None:
        for group, protocol in GROUPS.items():
            for name, obj in discover(group):
                key = (group, name)
                if key in self._by_key:
                    logger.warning(
                        "duplicate plugin name %r in group %s — keeping first, ignoring duplicate",
                        _safe_label(name),
                        _safe_label(group),
                    )
                    continue
                if _passes_protocol(obj, protocol):
                    self._by_key[key] = PluginInfo(
                        group=group,
                        name=name,
                        module=_module_path(obj),
                        status="loaded",
                        error=None,
                        obj=obj,
                    )
                else:
                    reason = f"does not satisfy Protocol {protocol.__name__}"
                    logger.warning(
                        "plugin %r in %s rejected at registration: %s",
                        _safe_label(name),
                        _safe_label(group),
                        reason,
                    )
                    self._by_key[key] = PluginInfo(
                        group=group,
                        name=name,
                        module=_module_path(obj),
                        status="error",
                        error=reason,
                        obj=None,
                    )

    def get(self, group: str, name: str) -> Any | None:
        info = self._by_key.get((group, name))
        if info is None or info.status != "loaded":
            return None
        return info.obj

    def list(self, group: str | None = None) -> list[PluginInfo]:
        if group is None:
            return list(self._by_key.values())
        return [info for (g, _), info in self._by_key.items() if g == group]

    def check(self, group: str, name: str) -> PluginInfo:
        info = self._by_key.get((group, name))
        if info is None:
            raise PluginValidationError(group=group, name=name, reason="not discovered")
        if info.status != "loaded":
            raise PluginValidationError(
                group=group, name=name, reason=info.error or "unknown error"
            )
        return info


@functools.lru_cache(maxsize=1)
def get_registry() -> PluginRegistry:
    """Return the process-wide plugin registry (built on first call).

    CPython CLI usage is single-threaded; concurrent first-call races
    are not guarded against. Callers in multi-threaded contexts (future
    web/TUI surfaces) should warm the cache from a single thread before
    fan-out, or add their own lock.
    """
    return PluginRegistry()
