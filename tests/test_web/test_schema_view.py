"""GET /api/schema (tree JSON) + GET /api/schema/erd (ERD fragment) tests (S-115).

The web stack lives in the optional ``[web]`` extra, so the module guards with
``pytest.importorskip("fastapi")`` before importing FastAPI symbols (mirrors the
sibling web tests). These endpoints are *read-only* over the in-memory
:class:`~dbsprout.web.workspace.Workspace` (``app.state.workspace``, S-111) that
``POST /api/connect`` (S-112) / ``POST /api/schema/load`` (S-113) populate:

* ``GET /api/schema`` returns the loaded schema as a tree-shaped JSON body
  (tables → columns/types/PKs/FKs); 404 friendly JSON when none is loaded.
* ``GET /api/schema/erd`` returns an HTMX ERD *fragment* rendering the workspace
  schema via the reused :func:`dbsprout.report.erd.build_erd_mermaid`; 200 +
  graceful empty state when none is loaded.

Tests seed ``app.state.workspace`` directly (a plain mutable object), exactly as
the connect tests inspect it; they never exercise the snapshot-backed
``GET /schema`` view (covered by ``test_erd.py``).
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


# ── fixtures / helpers ────────────────────────────────────────────────


def _make_app(tmp_path: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=tmp_path / "state.db")


def _small_schema() -> DatabaseSchema:
    """A 2-table schema: ``orders`` references ``users``."""
    users = TableSchema(
        name="users",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name="email", data_type=ColumnType.VARCHAR, unique=True, nullable=False),
        ],
        primary_key=["id"],
    )
    orders = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name="user_id", data_type=ColumnType.INTEGER),
            ColumnSchema(name="total", data_type=ColumnType.DECIMAL),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(
                columns=["user_id"], ref_table="users", ref_columns=["id"], on_delete="CASCADE"
            ),
        ],
    )
    return DatabaseSchema(tables=[users, orders], dialect="sqlite")


def _seed(app: FastAPI, schema: DatabaseSchema, source: str | None = None) -> None:
    """Load *schema* into the app's workspace, as connect/load would."""
    app.state.workspace.set_schema(schema)
    if source is not None:
        app.state.workspace.set_source(source)


def _table(body: dict, name: str) -> dict:
    """Pluck a single table entry out of the tree body by name."""
    return next(t for t in body["tables"] if t["name"] == name)


# ── GET /api/schema — loaded ───────────────────────────────────────────


def test_get_schema_returns_tree_for_loaded_schema(tmp_path: Path) -> None:
    """AC: returns the workspace schema as a tree (tables, columns, PKs, FKs)."""
    app = _make_app(tmp_path)
    _seed(app, _small_schema())
    resp = TestClient(app).get("/api/schema")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["table_count"] == 2
    assert body["dialect"] == "sqlite"
    assert {t["name"] for t in body["tables"]} == {"users", "orders"}


def test_get_schema_orders_columns_and_fk(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app, _small_schema())
    body = TestClient(app).get("/api/schema").json()
    orders = _table(body, "orders")
    assert orders["primary_key"] == ["id"]
    col_names = {c["name"] for c in orders["columns"]}
    assert {"id", "user_id", "total"} <= col_names
    user_id = next(c for c in orders["columns"] if c["name"] == "user_id")
    assert user_id["data_type"] == "integer"
    fk = orders["foreign_keys"][0]
    assert fk["ref_table"] == "users"
    assert fk["columns"] == ["user_id"]
    assert fk["ref_columns"] == ["id"]
    assert fk["on_delete"] == "CASCADE"


def test_get_schema_users_column_flags(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app, _small_schema())
    body = TestClient(app).get("/api/schema").json()
    users = _table(body, "users")
    id_col = next(c for c in users["columns"] if c["name"] == "id")
    email_col = next(c for c in users["columns"] if c["name"] == "email")
    assert id_col["primary_key"] is True
    assert email_col["primary_key"] is False
    assert email_col["unique"] is True
    assert email_col["nullable"] is False


def test_get_schema_echoes_redacted_source_no_password(tmp_path: Path) -> None:
    """AC: ``source`` is echoed; it is already redacted, so no password leaks."""
    app = _make_app(tmp_path)
    _seed(app, _small_schema(), source="db: sqlite:///x.db")
    resp = TestClient(app).get("/api/schema")
    assert resp.json()["source"] == "db: sqlite:///x.db"
    assert "password" not in resp.text.lower()


# ── GET /api/schema — empty state ──────────────────────────────────────


def test_get_schema_404_when_no_schema_loaded(tmp_path: Path) -> None:
    """AC: friendly 404 JSON when no schema is loaded (never a traceback)."""
    resp = TestClient(_make_app(tmp_path)).get("/api/schema")
    assert resp.status_code == 404, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert "connect" in detail.lower() or "load" in detail.lower()
    assert "Traceback" not in resp.text


# ── GET /api/schema/erd — loaded ───────────────────────────────────────


def test_schema_erd_fragment_renders_erdiagram(tmp_path: Path) -> None:
    """AC: ERD review of the loaded workspace schema via reused build_erd_mermaid."""
    app = _make_app(tmp_path)
    _seed(app, _small_schema())
    resp = TestClient(app).get("/api/schema/erd")
    assert resp.status_code == 200, resp.text
    body = resp.text
    assert "erDiagram" in body
    assert "users" in body
    assert "orders" in body
    assert 'class="mermaid"' in body


def test_schema_erd_fragment_has_detail_blob_and_click_handler(tmp_path: Path) -> None:
    """The fragment reuses the views/erd.py detail blob + click handler hooks."""
    app = _make_app(tmp_path)
    _seed(app, _small_schema())
    body = TestClient(app).get("/api/schema/erd").text
    assert "erd-table-data" in body
    assert "erdTableClick" in body
    assert "mermaid" in body.lower()


def test_schema_erd_is_a_fragment_not_full_page(tmp_path: Path) -> None:
    """S-117 hx-gets this into a panel — it must be a fragment, not a full page."""
    app = _make_app(tmp_path)
    _seed(app, _small_schema())
    body = TestClient(app).get("/api/schema/erd").text.lower()
    assert "<html" not in body
    assert "<!doctype" not in body


# ── GET /api/schema/erd — empty state ──────────────────────────────────


def test_schema_erd_fragment_200_empty_state_when_no_schema(tmp_path: Path) -> None:
    """AC: graceful 200 empty-state fragment (HTMX swap), not an error."""
    resp = TestClient(_make_app(tmp_path)).get("/api/schema/erd")
    assert resp.status_code == 200, resp.text
    assert "no schema" in resp.text.lower()


# ── router registration / seam ─────────────────────────────────────────


def test_schema_router_importable() -> None:
    from fastapi import APIRouter  # noqa: PLC0415

    from dbsprout.web.routers.schema import schema_router  # noqa: PLC0415

    assert isinstance(schema_router, APIRouter)


def test_schema_routes_registered_on_app(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/api/schema" in paths
    assert "/api/schema/erd" in paths


# ── lazy-import contract ───────────────────────────────────────────────


def test_schema_router_no_eager_generation_import() -> None:
    """Importing the router must not pull the orchestrator / core service."""
    probe = (
        "import sys\n"
        "import dbsprout.web.routers.schema  # noqa: F401\n"
        "bad = [m for m in ('dbsprout.generate.orchestrator', 'dbsprout.core.service')"
        " if m in sys.modules]\n"
        "print(bad)\n"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, trusted interpreter
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == "[]", (
        "importing dbsprout.web.routers.schema eagerly imported generation/core "
        f"modules: {result.stdout.strip()}"
    )
