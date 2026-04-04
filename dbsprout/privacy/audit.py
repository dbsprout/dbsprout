"""Append-only audit logging for LLM interactions.

Records every LLM call (and cache hit) as a JSON Lines entry in
``.dbsprout/audit.log``. The log is never truncated or overwritten.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class AuditEvent(BaseModel):
    """A single audit log entry for an LLM interaction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: str
    provider: str
    model: str | None = None
    privacy_tier: str | None = None
    schema_hash: str | None = None
    tokens_sent: int = 0
    tokens_received: int = 0
    cost_estimate: float = 0.0
    cached: bool = False
    duration_seconds: float | None = None


_DEFAULT_PATH = Path(".dbsprout/audit.log")


class AuditLog:
    """Append-only audit log backed by a JSON Lines file.

    Each ``record()`` call appends one JSON line. The file is
    created on the first write, not on initialization.
    """

    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self._path = path

    def record(self, event: AuditEvent) -> None:
        """Append an audit event as a single JSON line."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = event.model_dump_json()
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read(self, limit: int | None = None) -> list[AuditEvent]:
        """Read audit events from the log.

        Parameters
        ----------
        limit:
            If provided, return only the *limit* most recent entries.

        Returns
        -------
        list[AuditEvent]
            Events in chronological order. Empty list if file is
            missing or empty.
        """
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").strip().split("\n")
        lines = [ln for ln in lines if ln]
        if not lines:
            return []
        events: list[AuditEvent] = []
        for ln in lines:
            try:
                events.append(AuditEvent.model_validate(json.loads(ln)))
            except (json.JSONDecodeError, ValueError):
                logger.warning("Skipping malformed audit log line: %s", ln[:80])
        if limit is not None:
            events = events[-limit:]
        return events
