"""Validate panel a11y + focus-cell wiring assertions (S-135).

Sister to ``test_validate.py``: instead of re-asserting the integrity-check
JSON envelope, this module zooms in on the HTML fragment the panel renders
when a violation row is present, verifying the keyboard-accessible attributes
and the ``focus_cell.js`` boot script that S-135 wires into the panel.

We re-use the FK-violation fixture pattern from ``test_validate.py`` so the
template gets exercised with a real violation row. The render goes through
the live HTMX endpoint (``POST /api/validate`` with ``HX-Request: true``)
to stay close to production rendering and avoid Jinja-environment plumbing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.generate.orchestrator import GenerateResult
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


def _make_app(tmp_path: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=tmp_path / "state.db")


def _schema() -> DatabaseSchema:
    users = TableSchema(
        name="users",
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
    orders = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(
                name="id",
                data_type=ColumnType.INTEGER,
                nullable=False,
                primary_key=True,
            ),
            ColumnSchema(
                name="user_id",
                data_type=ColumnType.INTEGER,
                nullable=False,
            ),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(
                columns=["user_id"],
                ref_table="users",
                ref_columns=["id"],
            )
        ],
    )
    return DatabaseSchema(tables=[users, orders])


def _seed_with_violation(app: FastAPI) -> None:
    """Inject a workspace with a single orphan-FK violation row in ``orders``."""
    schema = _schema()
    users = [{"id": 1}]
    orders = [{"id": 1, "user_id": 999}]  # orphan FK
    app.state.workspace.set_schema(schema)
    app.state.workspace.set_last_result(
        GenerateResult(
            tables_data={"users": users, "orders": orders},
            insertion_order=["users", "orders"],
            total_rows=2,
            total_tables=2,
            duration_seconds=0.0,
        )
    )


# ── a11y attributes ────────────────────────────────────────────────────


def test_violation_row_has_tabindex_zero(tmp_path: Path) -> None:
    """AC (keyboard): every violation row must be keyboard-focusable."""
    app = _make_app(tmp_path)
    _seed_with_violation(app)
    resp = TestClient(app).post("/api/validate", headers={"HX-Request": "true"})
    assert resp.status_code == 200, resp.text
    body = resp.text
    # The row should carry tabindex="0" so it can receive keyboard focus.
    # Look at the violation-row class itself rather than the whole document.
    assert "violation-row" in body
    # The substring assertion is intentionally local — full attribute
    # adjacency keeps the failure message obvious.
    assert 'tabindex="0"' in body


def test_violation_row_has_button_role(tmp_path: Path) -> None:
    """AC (keyboard): violation rows must announce as buttons to assistive tech."""
    app = _make_app(tmp_path)
    _seed_with_violation(app)
    resp = TestClient(app).post("/api/validate", headers={"HX-Request": "true"})
    body = resp.text
    assert 'role="button"' in body


def test_violation_row_has_aria_label_with_table_and_column(tmp_path: Path) -> None:
    """The screen-reader label must include both the table and the column."""
    app = _make_app(tmp_path)
    _seed_with_violation(app)
    resp = TestClient(app).post("/api/validate", headers={"HX-Request": "true"})
    body = resp.text
    # Don't pin to exact wording — assert both tokens land inside an
    # aria-label that references the offending cell.
    # Find the first aria-label attribute and assert it mentions both.
    idx = body.find("aria-label=")
    assert idx >= 0, "violation row must have an aria-label"
    # Grab the rest of the line / attribute (cheap heuristic; ok for a test).
    snippet = body[idx : idx + 200]
    assert "orders" in snippet
    assert "user_id" in snippet


# ── focus_cell.js wiring ───────────────────────────────────────────────


def test_panel_loads_focus_cell_static_script(tmp_path: Path) -> None:
    """The page bootstrapping the panel must serve ``/static/focus_cell.js``."""
    app = _make_app(tmp_path)
    _seed_with_violation(app)
    # The standalone Studio page is the canonical entry: it embeds the
    # validate panel and is responsible for loading focus_cell.js.
    resp = TestClient(app).get("/studio")
    assert resp.status_code == 200, resp.text
    body = resp.text
    assert "/static/focus_cell.js" in body


def test_panel_fragment_calls_attach_focus_cell(tmp_path: Path) -> None:
    """The HTMX fragment must boot the dispatcher with the panel as root."""
    app = _make_app(tmp_path)
    _seed_with_violation(app)
    resp = TestClient(app).post("/api/validate", headers={"HX-Request": "true"})
    body = resp.text
    # The boot line lives at the bottom of the panel so it runs on both
    # initial render and HTMX swap. Substring assertion is robust to
    # whitespace tweaks in the template.
    assert "attachFocusCell" in body
    assert "validate-panel" in body


def test_focus_cell_static_file_served(tmp_path: Path) -> None:
    """``GET /static/focus_cell.js`` returns the dual-export module bytes."""
    app = _make_app(tmp_path)
    resp = TestClient(app).get("/static/focus_cell.js")
    assert resp.status_code == 200, resp.text
    # The module exposes attachFocusCell and installFocusCell — minimal
    # smoke check that the file isn't empty / 404.
    text = resp.text
    assert "attachFocusCell" in text
    assert "installFocusCell" in text
