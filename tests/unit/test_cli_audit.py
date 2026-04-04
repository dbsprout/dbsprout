"""Tests for dbsprout audit CLI command."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.privacy.audit import AuditEvent, AuditLog

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class TestAuditHelp:
    def test_help_output(self) -> None:
        result = runner.invoke(app, ["audit", "--help"])
        output = _strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--last" in output or "audit" in output.lower()


class TestAuditShowEmpty:
    def test_empty_log(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["audit"])
        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "no audit" in output.lower() or "0" in output


class TestAuditShowEntries:
    def test_displays_entries(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / ".dbsprout" / "audit.log"
        audit = AuditLog(path=log_path)
        audit.record(
            AuditEvent(
                timestamp="2026-04-03T12:00:00Z",
                provider="cloud",
                model="gpt-4o-mini",
                privacy_tier="redacted",
                tokens_sent=100,
                cost_estimate=0.001,
            )
        )
        result = runner.invoke(app, ["audit"])
        output = _strip_ansi(result.output)
        assert result.exit_code == 0
        assert "cloud" in output
        assert "gpt-4o" in output

    def test_last_n(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / ".dbsprout" / "audit.log"
        audit = AuditLog(path=log_path)
        for i in range(5):
            audit.record(
                AuditEvent(
                    timestamp=f"2026-04-03T12:0{i}:00Z",
                    provider=f"provider_{i}",
                )
            )
        result = runner.invoke(app, ["audit", "--last", "2"])
        output = _strip_ansi(result.output)
        assert result.exit_code == 0
        assert "provider_3" in output
        assert "provider_4" in output
        assert "provider_0" not in output
