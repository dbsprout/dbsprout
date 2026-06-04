"""GET /api/spec (DataSpec JSON) tests (S-118; JSON-only since P1c-5).

The web stack lives in the optional ``[web]`` extra, so this module guards
imports with ``pytest.importorskip("fastapi")`` before pulling the FastAPI
symbols (mirrors the sibling web tests). ``GET /api/spec`` is a *read-only*
endpoint over the in-memory :class:`~dbsprout.web.workspace.Workspace`
(``app.state.workspace``, S-111):

* returns the active ``DataSpec`` as JSON — building a heuristic spec lazily
  when none has been cached on the workspace yet — or a 409
  ``{code: "NO_SCHEMA"}`` envelope when no schema is loaded.

The legacy HTMX ``Accept: text/html`` spec-grid fragment was removed in the
P1c-5 cutover; the endpoint now returns JSON unconditionally.
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
    ForeignKeySchema,
    TableSchema,
)
from dbsprout.spec.models import DataSpec, GeneratorConfig, TableSpec

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
            ColumnSchema(name="email", data_type=ColumnType.VARCHAR, max_length=255),
        ],
        primary_key=["id"],
    )
    orders = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name="user_id", data_type=ColumnType.INTEGER),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(
                columns=("user_id",),
                ref_table="users",
                ref_columns=("id",),
            ),
        ],
    )
    return DatabaseSchema(tables=[users, orders])


def _seed_workspace(app: FastAPI, schema: DatabaseSchema) -> None:
    app.state.workspace.set_schema(schema)


# ── JSON path ─────────────────────────────────────────────────────────


def test_get_spec_no_schema_returns_409_with_code(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)

    response = client.get("/api/spec")

    assert response.status_code == 409
    body = response.json()
    detail = body["detail"]
    # FastAPI puts the dict directly under ``detail`` when we raise with a dict
    assert isinstance(detail, dict)
    assert detail["code"] == "NO_SCHEMA"
    assert "message" in detail
    assert "schema" in detail["message"].lower()


def test_get_spec_builds_heuristic_when_absent(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    # Sanity: workspace has no spec cached yet.
    assert app.state.workspace.get_spec() is None
    client = TestClient(app)

    response = client.get("/api/spec")

    assert response.status_code == 200
    body = response.json()
    table_names = [t["table_name"] for t in body["tables"]]
    assert table_names == ["users", "orders"]
    # Each table carries column configs.
    users_spec = body["tables"][0]
    assert set(users_spec["columns"].keys()) == {"id", "email"}
    # Spec is now cached on the workspace.
    assert app.state.workspace.get_spec() is not None


def test_get_spec_returns_cached_on_second_call(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)

    # First call builds + caches.
    first = client.get("/api/spec")
    assert first.status_code == 200

    # Replace the cached spec with a sentinel so we can prove the second call
    # returns the cached object rather than rebuilding from the schema.
    sentinel = DataSpec(
        tables=[
            TableSpec(
                table_name="users",
                row_count=999,
                columns={
                    "id": GeneratorConfig(provider="sentinel.marker", method="sentinel"),
                },
            ),
        ],
        model_used="cache-sentinel",
    )
    app.state.workspace.set_spec(sentinel)

    second = client.get("/api/spec")
    body = second.json()
    assert second.status_code == 200
    assert body["model_used"] == "cache-sentinel"
    assert body["tables"][0]["row_count"] == 999
    assert body["tables"][0]["columns"]["id"]["provider"] == "sentinel.marker"


def test_get_spec_includes_row_count_and_provider_per_column(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)

    response = client.get("/api/spec")
    body = response.json()

    for table in body["tables"]:
        assert "row_count" in table
        assert isinstance(table["row_count"], int)
        assert table["row_count"] >= 1
        for col_name, col_cfg in table["columns"].items():
            assert "provider" in col_cfg, f"missing provider on {table['table_name']}.{col_name}"
            assert "method" in col_cfg
            assert "params" in col_cfg


def test_get_spec_returns_json_even_with_accept_text_html(tmp_path: Path) -> None:
    """JSON-only since P1c-5: an HTML ``Accept`` no longer selects a fragment."""
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)

    response = client.get("/api/spec", headers={"Accept": "text/html"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert [t["table_name"] for t in response.json()["tables"]] == ["users", "orders"]
