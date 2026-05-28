"""GET /api/spec (DataSpec JSON + HTMX fragment) tests (S-118).

The web stack lives in the optional ``[web]`` extra, so this module guards
imports with ``pytest.importorskip("fastapi")`` before pulling the FastAPI
symbols (mirrors the sibling web tests). ``GET /api/spec`` is a *read-only*
endpoint over the in-memory :class:`~dbsprout.web.workspace.Workspace`
(``app.state.workspace``, S-111):

* JSON (default ``Accept``): returns the active ``DataSpec`` — building a
  heuristic spec lazily when none has been cached on the workspace yet — or a
  409 ``{code: "NO_SCHEMA"}`` envelope when no schema is loaded.
* HTML (``Accept: text/html``): renders the spec grid fragment; on missing
  schema returns 200 with a friendly empty-state body so HTMX swaps cleanly.

The fragment exposes a stable shape for the upcoming S-119 edit endpoint:
per-column rows with ``data-table`` / ``data-column`` and a method pill button
carrying ``data-method`` / ``data-provider``.
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


# ── HTML fragment path ────────────────────────────────────────────────


def _empty_schema() -> DatabaseSchema:
    return DatabaseSchema(tables=[])


def _wide_schema(num_columns: int = 120) -> DatabaseSchema:
    """Synthetic single-table schema with *num_columns* INTEGER columns."""
    columns = [ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True)]
    columns.extend(
        ColumnSchema(name=f"col_{i:03d}", data_type=ColumnType.INTEGER)
        for i in range(num_columns - 1)
    )
    return DatabaseSchema(
        tables=[TableSchema(name="wide", columns=columns, primary_key=["id"])],
    )


def _long_name_schema() -> DatabaseSchema:
    long_name = "a_very_long_column_name_that_exceeds_thirty_two_chars"
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="t",
                columns=[
                    ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
                    ColumnSchema(name=long_name, data_type=ColumnType.VARCHAR),
                ],
                primary_key=["id"],
            ),
        ],
    )


def test_get_spec_html_fragment_renders_grid(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)

    response = client.get("/api/spec", headers={"Accept": "text/html"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "data-spec-grid" in body
    # Per-column rows are addressable by ``data-column="table.column"``.
    assert 'data-column="users.id"' in body
    assert 'data-column="users.email"' in body
    assert 'data-column="orders.user_id"' in body
    # Each row carries a method-pill button with ``data-method`` + ``data-provider``.
    assert "data-method=" in body
    assert "data-provider=" in body
    # And the pill is a real button (S-119 will wire its click behaviour).
    assert 'type="button"' in body


def test_get_spec_html_empty_schema(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _empty_schema())
    client = TestClient(app)

    response = client.get("/api/spec", headers={"Accept": "text/html"})

    assert response.status_code == 200
    body = response.text
    # The wrapper is always present, but there are no table sections; the
    # template surfaces a friendly "no tables" message.
    assert "data-spec-grid" in body
    assert "data-empty-tables" in body


def test_get_spec_html_no_schema_returns_empty_state(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)

    response = client.get("/api/spec", headers={"Accept": "text/html"})

    assert response.status_code == 200
    body = response.text
    assert "data-empty-state" in body
    # Hint nudges the user toward the connect / upload routes.
    assert "connect" in body.lower() or "upload" in body.lower()


def test_get_spec_html_truncation_attrs_for_long_names(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _long_name_schema())
    client = TestClient(app)

    response = client.get("/api/spec", headers={"Accept": "text/html"})
    body = response.text

    long_name = "a_very_long_column_name_that_exceeds_thirty_two_chars"
    # Native tooltip via ``title`` attribute.
    assert f'title="{long_name}"' in body
    # And the truncate class is on the column-name span / cell.
    assert "truncate" in body


def test_get_spec_html_handles_100_plus_columns(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _wide_schema(120))
    client = TestClient(app)

    response = client.get("/api/spec", headers={"Accept": "text/html"})
    body = response.text

    assert response.status_code == 200
    # Every column rendered.
    for i in range(119):
        assert f'data-column="wide.col_{i:03d}"' in body
    assert 'data-column="wide.id"' in body
    # Wrapper element rendered exactly once.
    assert body.count("data-spec-grid") == 1


def test_get_spec_html_accept_with_json_preference_returns_json(tmp_path: Path) -> None:
    """Sanity for content negotiation: JSON wins when listed before HTML."""
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)

    response = client.get(
        "/api/spec",
        headers={"Accept": "application/json, text/html;q=0.5"},
    )

    assert response.headers["content-type"].startswith("application/json")
    assert response.status_code == 200
