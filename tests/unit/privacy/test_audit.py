"""Tests for dbsprout.privacy.audit — append-only audit logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import ValidationError

from dbsprout.privacy.audit import AuditEvent, AuditLog

if TYPE_CHECKING:
    from pathlib import Path

import pytest


class TestAuditEvent:
    """AuditEvent Pydantic model."""

    def test_required_fields(self) -> None:
        event = AuditEvent(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            provider="cloud",
        )
        assert event.provider == "cloud"

    def test_all_fields(self) -> None:
        event = AuditEvent(
            timestamp="2026-04-03T12:00:00Z",
            provider="cloud",
            model="gpt-4o-mini",
            privacy_tier="redacted",
            schema_hash="abc123",
            tokens_sent=100,
            tokens_received=50,
            cost_estimate=0.001,
            cached=False,
            duration_seconds=1.5,
        )
        assert event.model == "gpt-4o-mini"
        assert event.tokens_sent == 100
        assert event.cost_estimate == 0.001

    def test_frozen(self) -> None:
        event = AuditEvent(timestamp="2026-04-03T12:00:00Z", provider="cloud")
        with pytest.raises(ValidationError):
            event.provider = "local"  # type: ignore[misc]

    def test_serializes_to_json(self) -> None:
        event = AuditEvent(
            timestamp="2026-04-03T12:00:00Z",
            provider="embedded",
            cached=True,
        )
        data = json.loads(event.model_dump_json())
        assert data["provider"] == "embedded"
        assert data["cached"] is True

    def test_defaults(self) -> None:
        event = AuditEvent(timestamp="2026-04-03T12:00:00Z", provider="cloud")
        assert event.model is None
        assert event.tokens_sent == 0
        assert event.tokens_received == 0
        assert event.cost_estimate == 0.0
        assert event.cached is False
        assert event.duration_seconds is None


class TestAuditLogRecord:
    """AuditLog.record — append-only writes."""

    def test_creates_file_on_first_record(self, tmp_path: Path) -> None:
        log_path = tmp_path / ".dbsprout" / "audit.log"
        audit = AuditLog(path=log_path)
        event = AuditEvent(timestamp="2026-04-03T12:00:00Z", provider="cloud")
        audit.record(event)
        assert log_path.exists()

    def test_appends_json_line(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.log"
        audit = AuditLog(path=log_path)
        e1 = AuditEvent(timestamp="2026-04-03T12:00:00Z", provider="cloud")
        e2 = AuditEvent(timestamp="2026-04-03T12:01:00Z", provider="embedded")
        audit.record(e1)
        audit.record(e2)
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["provider"] == "cloud"
        assert json.loads(lines[1])["provider"] == "embedded"

    def test_append_only(self, tmp_path: Path) -> None:
        """Second write doesn't overwrite first."""
        log_path = tmp_path / "audit.log"
        audit = AuditLog(path=log_path)
        e1 = AuditEvent(timestamp="2026-04-03T12:00:00Z", provider="first")
        e2 = AuditEvent(timestamp="2026-04-03T12:01:00Z", provider="second")
        audit.record(e1)
        audit.record(e2)
        content = log_path.read_text()
        assert "first" in content
        assert "second" in content


class TestAuditLogRead:
    """AuditLog.read — reading events back."""

    def test_read_all(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.log"
        audit = AuditLog(path=log_path)
        for i in range(3):
            audit.record(AuditEvent(timestamp=f"2026-04-03T12:0{i}:00Z", provider=f"p{i}"))
        events = audit.read()
        assert len(events) == 3
        assert events[0].provider == "p0"
        assert events[2].provider == "p2"

    def test_read_empty_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.log"
        audit = AuditLog(path=log_path)
        events = audit.read()
        assert events == []

    def test_read_missing_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "nonexistent.log"
        audit = AuditLog(path=log_path)
        events = audit.read()
        assert events == []

    def test_read_limit(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.log"
        audit = AuditLog(path=log_path)
        for i in range(5):
            audit.record(AuditEvent(timestamp=f"2026-04-03T12:0{i}:00Z", provider=f"p{i}"))
        events = audit.read(limit=2)
        assert len(events) == 2
        assert events[0].provider == "p3"
        assert events[1].provider == "p4"
