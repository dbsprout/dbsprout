"""Discover dbsprout plugins via ``importlib.metadata.entry_points``."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


def _safe_label(value: object) -> str:
    """Render *value* as a single-line, control-char-free string.

    Entry-point ``name``s come from third-party package metadata and are
    not validated by the importlib runtime — a hostile (or buggy)
    package can declare a name that contains newlines or terminal
    escape codes. Sanitising before logging closes the log-injection
    surface (CWE-117) and keeps the warning lines greppable.
    """
    text = str(value)
    return "".join(ch if ch.isprintable() and ch not in ("\n", "\r") else "?" for ch in text)


_ALLOWED_GROUPS: frozenset[str] = frozenset(
    {
        "dbsprout.parsers",
        "dbsprout.generators",
        "dbsprout.outputs",
        "dbsprout.llm_providers",
        "dbsprout.migration_frameworks",
        "dbsprout.train_extractors",
    }
)


def discover(group: str) -> Iterator[tuple[str, Any]]:
    """Yield ``(name, obj)`` for every loadable entry point in *group*.

    Broken entry points (import errors, missing attributes, etc.) are
    logged as warnings and skipped. This function never raises.

    Only the five DBSprout entry-point groups are accepted; arbitrary
    groups are rejected with ``ValueError`` to prevent callers (or
    future CLI surfaces that take group names as input) from
    enumerating unrelated installed packages.
    """
    if group not in _ALLOWED_GROUPS:
        raise ValueError(f"unknown plugin group: {group!r}")
    for ep in entry_points(group=group):
        try:
            obj = ep.load()
        except Exception as exc:
            logger.warning(
                "plugin load failed: group=%s name=%s error=%s: %s",
                _safe_label(group),
                _safe_label(ep.name),
                type(exc).__name__,
                _safe_label(exc),
            )
            continue
        yield ep.name, obj
