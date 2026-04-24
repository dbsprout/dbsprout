"""Discover dbsprout plugins via ``importlib.metadata.entry_points``."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


def discover(group: str) -> Iterator[tuple[str, Any]]:
    """Yield ``(name, obj)`` for every loadable entry point in *group*.

    Broken entry points (import errors, missing attributes, etc.) are
    logged as warnings and skipped. This function never raises.
    """
    for ep in entry_points(group=group):
        try:
            obj = ep.load()
        except Exception as exc:
            logger.warning(
                "plugin load failed: group=%s name=%s error=%s: %s",
                group,
                ep.name,
                type(exc).__name__,
                exc,
            )
            continue
        yield ep.name, obj
