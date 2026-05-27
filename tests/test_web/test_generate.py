"""POST /api/generate background-job submit endpoint tests (S-124).

The web stack lives in the optional ``[web]`` extra, so the module guards with
``pytest.importorskip("fastapi")`` before importing FastAPI symbols (mirrors the
sibling web tests). This is the first route that *starts work*: it composes the
S-106 core service, the S-108 job manager, and the S-111 workspace into a
non-blocking submit that returns a ``job_id`` immediately.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


def _make_app(state_db: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=state_db)


def _temp_sqlite(tmp_path: Path) -> str:
    """Create a 2-table sqlite DB and return its ``sqlite:///`` URL."""
    db_path = tmp_path / "gen.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute(
            "CREATE TABLE posts (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id))"
        )
        conn.commit()
    finally:
        conn.close()
    return f"sqlite:///{db_path}"


def _load_schema(app: FastAPI, tmp_path: Path) -> None:
    """Connect the app's workspace to a temp sqlite DB (reuses /api/connect)."""
    TestClient(app).post("/api/connect", json={"url": _temp_sqlite(tmp_path)})


# ── router registration / seam ─────────────────────────────────────────


def test_generate_router_is_importable() -> None:
    from fastapi import APIRouter  # noqa: PLC0415

    from dbsprout.web.routers.generate import generate_router  # noqa: PLC0415

    assert isinstance(generate_router, APIRouter)


# ── happy path: submit returns a job_id ────────────────────────────────


def test_generate_returns_job_id(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)
    resp = TestClient(app).post("/api/generate", json={"seed": 7})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["job_id"], str)
    assert body["job_id"]


def test_generate_defaults_when_body_empty(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)
    resp = TestClient(app).post("/api/generate", json={})
    assert resp.status_code == 200, resp.text
    assert resp.json()["job_id"]


# ── no-schema guard + input validation ─────────────────────────────────


def test_generate_without_schema_is_4xx(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")  # no schema loaded
    resp = TestClient(app).post("/api/generate", json={})
    assert 400 <= resp.status_code < 500, resp.text
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert "schema" in detail.lower()
    assert "Traceback" not in resp.text


def test_generate_unknown_engine_is_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)
    resp = TestClient(app).post("/api/generate", json={"engine": "bogus"})
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert "bogus" in detail.lower() or "engine" in detail.lower()


def test_generate_extra_field_is_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)
    resp = TestClient(app).post("/api/generate", json={"rows": 10})
    assert resp.status_code == 422


def test_generate_negative_seed_is_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)
    resp = TestClient(app).post("/api/generate", json={"seed": -1})
    assert resp.status_code == 422
