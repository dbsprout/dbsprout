"""GET /api/samples + POST /api/schema/sample (P1a)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path


def _client(tmp_path: Path) -> TestClient:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return TestClient(create_app(state_db_path=tmp_path / "state.db"))


def test_list_samples(tmp_path: Path) -> None:
    resp = _client(tmp_path).get("/api/samples")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    names = {s["name"] for s in body["samples"]}
    assert {"ecommerce", "saas"} <= names
    one = next(s for s in body["samples"] if s["name"] == "ecommerce")
    assert one["table_count"] > 0
    assert one["title"]
    assert one["description"]
    assert one["dialect"]


def test_load_sample_into_workspace(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post("/api/schema/sample", json={"name": "ecommerce"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["table_count"] > 0
    assert body["source"].startswith("sample:")
    assert client.get("/api/schema").json()["table_count"] == body["table_count"]


def test_load_unknown_sample_404(tmp_path: Path) -> None:
    resp = _client(tmp_path).post("/api/schema/sample", json={"name": "nope"})
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"]  # typed envelope
