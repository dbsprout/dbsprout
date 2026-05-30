"""Tests for the lightweight connection-probe primitive (P1a)."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from dbsprout.core.service import ConnectionProbe, probe_connection

if TYPE_CHECKING:
    from pathlib import Path


def _two_table_db(tmp_path: Path) -> str:
    db = tmp_path / "probe.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE a (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE b (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    return f"sqlite:///{db}"


def test_probe_reports_dialect_and_table_count(tmp_path: Path) -> None:
    probe = probe_connection(_two_table_db(tmp_path))
    assert isinstance(probe, ConnectionProbe)
    assert probe.dialect == "sqlite"
    assert probe.table_count == 2
    assert probe.latency_ms >= 0
    assert probe.server_version  # non-empty string


def test_probe_unsupported_dialect_raises_valueerror() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        probe_connection("oracle://user:pw@host/db")


def test_probe_bad_url_raises() -> None:
    with pytest.raises(Exception):  # noqa: B017,PT011 - broad on purpose; caller narrows
        probe_connection("not-a-url")
