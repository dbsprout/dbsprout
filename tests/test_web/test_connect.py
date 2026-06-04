"""POST /api/connect live-introspection endpoint tests (S-112).

The web stack lives in the optional ``[web]`` extra, so the module guards with
``pytest.importorskip("fastapi")`` before importing FastAPI symbols (mirrors the
sibling web tests). The endpoint is the first read-WRITE JSON API in the
dashboard: it introspects a live DB via the S-106 core-service facade, stores the
schema + redacted target in the S-111 workspace, and returns a JSON summary —
failing with a friendly, credential-scrubbed 4xx (FR-009), never a traceback.
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
    db_path = tmp_path / "connect.db"
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


# ── success path ──────────────────────────────────────────────────────


def test_connect_returns_schema_summary(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    url = _temp_sqlite(tmp_path)
    resp = TestClient(app).post("/api/connect", json={"url": url})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["table_count"] == 2
    assert set(body["tables"]) == {"users", "posts"}
    assert body["dialect"] == "sqlite"


def test_connect_stores_schema_and_redacted_target_in_workspace(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    url = _temp_sqlite(tmp_path)
    TestClient(app).post("/api/connect", json={"url": url})
    ws = app.state.workspace
    schema = ws.get_schema()
    assert schema is not None
    assert set(schema.table_names()) == {"users", "posts"}
    # sqlite URL has no password → redacted target is the URL unchanged.
    assert ws.redacted_target == url
    source = ws.get_source()
    assert source is not None
    assert source.startswith("db: ")


def test_connect_stores_raw_url_privately_never_echoed(tmp_path: Path) -> None:
    """The raw URL is stored only via the private target; the source descriptor
    holds the redacted form, and the response body never contains a password."""
    app = _make_app(tmp_path / "state.db")
    # A sqlite URL has no creds, but the descriptor must still be the redacted
    # form (here identical). This guards the wiring, not masking (covered below).
    url = _temp_sqlite(tmp_path)
    resp = TestClient(app).post("/api/connect", json={"url": url})
    assert resp.status_code == 200
    assert "password" not in resp.text.lower()


# ── input validation (Pydantic boundary) ──────────────────────────────


def test_connect_missing_url_is_422(tmp_path: Path) -> None:
    resp = TestClient(_make_app(tmp_path / "state.db")).post("/api/connect", json={})
    assert resp.status_code == 422


def test_connect_blank_url_is_422(tmp_path: Path) -> None:
    resp = TestClient(_make_app(tmp_path / "state.db")).post("/api/connect", json={"url": "   "})
    assert resp.status_code == 422


def test_connect_extra_field_is_422(tmp_path: Path) -> None:
    """``extra='forbid'`` rejects unexpected keys at the boundary."""
    resp = TestClient(_make_app(tmp_path / "state.db")).post(
        "/api/connect", json={"url": "sqlite:///x.db", "rows": 10}
    )
    assert resp.status_code == 422


# ── friendly errors (FR-009) ───────────────────────────────────────────


def test_connect_unsupported_dialect_returns_friendly_4xx(tmp_path: Path) -> None:
    resp = TestClient(_make_app(tmp_path / "state.db")).post(
        "/api/connect", json={"url": "redis://localhost:6379/0"}
    )
    assert 400 <= resp.status_code < 500
    detail = resp.json()["detail"]
    # S-116: envelope shape — {code, message, correlation_id, hint?}.
    assert isinstance(detail, dict)
    assert detail["code"] == "UNKNOWN_DIALECT"
    assert "correlation_id" in detail
    # JSON error envelope, never an HTML traceback page.
    assert "Traceback" not in resp.text


def test_connect_malformed_url_returns_friendly_4xx(tmp_path: Path) -> None:
    resp = TestClient(_make_app(tmp_path / "state.db")).post(
        "/api/connect", json={"url": "not a url at all"}
    )
    # Exactly 400 (the friendly-error status) — distinguishes a real handled
    # error from an accidental 404 (missing route) or 422 (validation).
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, dict)
    # SQLAlchemy raises ArgumentError for unparseable URLs → MALFORMED_URL;
    # but if the URL parses but the dialect is unknown we land on UNKNOWN_DIALECT.
    # Either way it's a caller-actionable 400 with the new envelope.
    assert detail["code"] in {"MALFORMED_URL", "UNKNOWN_DIALECT", "CONN_REFUSED"}
    assert "correlation_id" in detail
    assert "Traceback" not in resp.text


def test_connect_error_redacts_credentials(tmp_path: Path) -> None:
    """A failing URL with credentials must never leak the password in the body."""
    app = _make_app(tmp_path / "state.db")
    # Postgres driver (psycopg2) is not installed in the test env → connect-time
    # failure; either way the password must be scrubbed from the response.
    resp = TestClient(app).post(
        "/api/connect",
        json={"url": "postgresql://alice:s3cretpw@badhost.invalid:5432/app"},
    )
    assert resp.status_code == 400, resp.text
    assert "s3cretpw" not in resp.text
    assert "Traceback" not in resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, dict)
    assert "code" in detail
    assert "correlation_id" in detail


def test_connect_failure_leaves_workspace_clean(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    TestClient(app).post(
        "/api/connect",
        json={"url": "postgresql://bob:hunter2@nope.invalid:5432/db"},
    )
    ws = app.state.workspace
    assert ws.get_schema() is None
    assert ws.redacted_target is None
    assert ws.get_source() is None


# ── router registration / seam ─────────────────────────────────────────


def test_connect_router_is_importable() -> None:
    from fastapi import APIRouter  # noqa: PLC0415

    from dbsprout.web.routers.connect import connect_router  # noqa: PLC0415

    assert isinstance(connect_router, APIRouter)


def test_connect_route_registered_on_app(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/api/connect" in paths


# ── S-116: typed envelope + HTMX + INTERNAL ─────────────────────────────


def test_connect_returns_typed_envelope_on_failure(tmp_path: Path) -> None:
    """S-116: every failure carries ``{code, message, correlation_id}``."""
    resp = TestClient(_make_app(tmp_path / "state.db")).post(
        "/api/connect", json={"url": "postgresql://u:p@unreachable.invalid:5432/db"}
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["code"] in {"CONN_REFUSED", "MISSING_DRIVER"}
    assert detail["message"]
    assert detail["correlation_id"]
    assert "Traceback" not in resp.text


def test_connect_htmx_request_returns_json_envelope(tmp_path: Path) -> None:
    """JSON-only since P1c-5: an ``HX-Request`` header no longer yields HTML."""
    resp = TestClient(_make_app(tmp_path / "state.db")).post(
        "/api/connect",
        json={"url": "redis://localhost:6379/0"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("application/json")
    detail = resp.json()["detail"]
    assert detail["code"] == "UNKNOWN_DIALECT"
    assert detail["correlation_id"]
    # No stack frame leaks into the body.
    assert "Traceback" not in resp.text


def test_connect_unexpected_exception_returns_internal_500(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S-116 AC: unclassified exceptions become INTERNAL/500 with correlation id."""

    def _boom(_source: object) -> object:
        raise RuntimeError("simulated catastrophic failure")

    monkeypatch.setattr("dbsprout.web.routers.connect.load_schema", _boom, raising=False)
    # Lazy-imported in the handler — also patch its source module just in case.
    monkeypatch.setattr("dbsprout.core.service.load_schema", _boom)

    resp = TestClient(_make_app(tmp_path / "state.db")).post(
        "/api/connect", json={"url": "sqlite:///:memory:"}
    )
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["code"] == "INTERNAL"
    assert detail["message"] == "Unexpected error"
    assert detail["correlation_id"]
    # The real exception text must NOT appear in the user-facing body.
    assert "simulated catastrophic failure" not in resp.text
    assert "Traceback" not in resp.text
