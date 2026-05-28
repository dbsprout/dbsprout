"""Spec grid focus-cell listener wiring (S-135).

Sister to ``test_spec_view.py``: this module verifies the spec grid fragment
boots the ``focus_cell.js`` listener so it can react to ``studio:focus-cell``
events dispatched by S-133's integrity report.

The grid is rendered as an HTMX fragment via ``GET /api/spec`` once a schema
has been loaded into the workspace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


def _make_app(tmp_path: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=tmp_path / "state.db")


def _two_table_schema() -> DatabaseSchema:
    users = TableSchema(
        name="users",
        columns=[
            ColumnSchema(
                name="id",
                data_type=ColumnType.INTEGER,
                nullable=False,
                primary_key=True,
            ),
            ColumnSchema(
                name="email",
                data_type=ColumnType.VARCHAR,
                nullable=False,
            ),
        ],
        primary_key=["id"],
    )
    orders = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(
                name="id",
                data_type=ColumnType.INTEGER,
                nullable=False,
                primary_key=True,
            ),
        ],
        primary_key=["id"],
    )
    return DatabaseSchema(tables=[users, orders])


def _seed_schema(app: FastAPI) -> None:
    app.state.workspace.set_schema(_two_table_schema())


# ── boot snippet wiring ────────────────────────────────────────────────


def test_spec_grid_fragment_boots_focus_cell_listener(tmp_path: Path) -> None:
    """The spec-grid fragment must call ``installFocusCell`` once it lands in the DOM."""
    app = _make_app(tmp_path)
    _seed_schema(app)
    resp = TestClient(app).get("/api/spec", headers={"Accept": "text/html"})
    assert resp.status_code == 200, resp.text
    body = resp.text
    # The fragment is responsible for self-bootstrapping the listener so
    # HTMX swaps re-wire it on every refresh.
    assert "installFocusCell" in body
    assert "spec-grid" in body


def test_spec_grid_boots_focus_cell_exactly_once(tmp_path: Path) -> None:
    """The boot snippet must appear exactly once in the rendered fragment."""
    app = _make_app(tmp_path)
    _seed_schema(app)
    resp = TestClient(app).get("/api/spec", headers={"Accept": "text/html"})
    body = resp.text
    assert body.count("installFocusCell(") == 1


def test_studio_page_loads_focus_cell_script(tmp_path: Path) -> None:
    """The Studio page (host of both the spec grid and the validate panel) must load the file."""
    app = _make_app(tmp_path)
    _seed_schema(app)
    resp = TestClient(app).get("/studio")
    assert resp.status_code == 200, resp.text
    assert "/static/focus_cell.js" in resp.text
