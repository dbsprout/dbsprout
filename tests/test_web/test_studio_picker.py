"""Studio method-picker UI surface tests (S-120).

These tests pin the *shape* the picker relies on without exercising any
JavaScript runtime:

* The pill button rendered by ``spec_row.html`` carries the data-* hooks
  the Alpine picker reads (table, column, method, provider, **dtype**).
* The Studio page renders the picker template (Alpine component) once,
  with a stable element id so the page-level handler can find it.

The browser-side interaction (open, fetch /api/generators, apply via
PUT /api/spec/tables/{t}/columns/{c}) is tested at the API level by
``test_generators_catalog.py`` + the existing ``test_spec_update.py`` —
this file only guarantees the DOM contract.
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


def _seed_workspace(app: FastAPI) -> None:
    """Drop a tiny schema on the workspace so /api/spec renders rows."""
    schema = DatabaseSchema(
        tables=[
            TableSchema(
                name="users",
                columns=[
                    ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
                    ColumnSchema(name="email", data_type=ColumnType.VARCHAR, max_length=255),
                ],
                primary_key=["id"],
            ),
        ]
    )
    app.state.workspace.set_schema(schema)


def test_studio_page_embeds_picker_component(tmp_path: Path) -> None:
    """Studio page must include the method-picker Alpine root."""
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/studio")
    assert resp.status_code == 200
    body = resp.text
    # Stable hooks the picker depends on — keep these in lockstep with
    # ``templates/studio/method_picker.html``.
    assert "method-picker" in body, "picker root id/data attr missing from studio page"
    assert "x-data" in body, "Alpine x-data binding expected on picker"


def test_spec_row_pill_carries_dtype_attribute(tmp_path: Path) -> None:
    """Pill must expose the column dtype so the picker can pre-filter."""
    app = _make_app(tmp_path)
    _seed_workspace(app)
    with TestClient(app) as client:
        resp = client.get("/api/spec", headers={"Accept": "text/html"})
    assert resp.status_code == 200
    body = resp.text
    # ``email`` is a VARCHAR column in the seeded schema.
    # The pill must carry data-dtype so the picker can filter on it.
    assert 'data-dtype="VARCHAR"' in body, body


def test_spec_row_pill_keeps_existing_data_hooks(tmp_path: Path) -> None:
    """S-118 + S-119 hooks must still be present (regression guard)."""
    app = _make_app(tmp_path)
    _seed_workspace(app)
    with TestClient(app) as client:
        resp = client.get("/api/spec", headers={"Accept": "text/html"})
    body = resp.text
    assert 'data-table="users"' in body
    assert 'data-column="email"' in body
    assert "data-method=" in body
    assert "data-provider=" in body
