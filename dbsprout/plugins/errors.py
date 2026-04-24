"""Plugin system error types."""

from __future__ import annotations


class PluginError(Exception):
    """Base error for plugin discovery and validation failures."""


class PluginValidationError(PluginError):
    """Raised when a plugin does not satisfy its Protocol."""

    def __init__(self, *, group: str, name: str, reason: str) -> None:
        self.group = group
        self.name = name
        self.reason = reason
        super().__init__(f"plugin {group}:{name} invalid — {reason}")
