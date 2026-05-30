"""POST /api/schema/paste (P1a)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path

_DDL = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);\n"


def _client(tmp_path: Path) -> TestClient:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return TestClient(create_app(state_db_path=tmp_path / "state.db"))


def test_paste_ddl_loads_schema(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post("/api/schema/paste", json={"text": _DDL})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "paste"
    assert body["tables"] == ["users"]
    assert client.get("/api/schema").json()["table_count"] == 1


def test_paste_empty_is_4xx(tmp_path: Path) -> None:
    resp = _client(tmp_path).post("/api/schema/paste", json={"text": "   "})
    assert resp.status_code in (400, 422)


def test_paste_garbage_is_parse_error(tmp_path: Path) -> None:
    resp = _client(tmp_path).post("/api/schema/paste", json={"text": "}{ not a schema"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"]  # typed envelope
