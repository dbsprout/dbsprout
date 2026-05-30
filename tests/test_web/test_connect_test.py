"""POST /api/connect/test (P1a)."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path


def _client(tmp_path: Path) -> TestClient:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return TestClient(create_app(state_db_path=tmp_path / "state.db"))


def _db(tmp_path: Path) -> str:
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE x (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    return f"sqlite:///{db}"


def test_connect_test_ok(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post("/api/connect/test", json={"url": _db(tmp_path)})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["dialect"] == "sqlite"
    assert body["table_count"] == 1
    assert "server_version" in body
    assert "latency_ms" in body


def test_connect_test_does_not_load_workspace(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.post("/api/connect/test", json={"url": _db(tmp_path)})
    assert client.get("/api/schema").status_code == 404  # probe must not populate workspace


def test_connect_test_bad_url_typed_error(tmp_path: Path) -> None:
    resp = _client(tmp_path).post("/api/connect/test", json={"url": "oracle://h/db"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"]  # typed envelope (UNKNOWN_DIALECT)
