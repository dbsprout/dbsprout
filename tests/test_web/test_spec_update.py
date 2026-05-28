"""``PUT /api/spec/tables/{t}/columns/{c}`` — edit a column GeneratorConfig (S-119).

The endpoint extends the S-118 spec router to support in-place edits:

* JSON body matches :class:`~dbsprout.spec.models.GeneratorConfig`; bad input
  yields ``422`` with field-level Pydantic errors.
* A referential-integrity guard
  (:func:`dbsprout.spec.constraints.check_column_update`) rejects changes that
  would break PK / FK invariants with ``409 CONSTRAINT_VIOLATION``.
* Missing schema → ``409 NO_SCHEMA`` (matches the S-118 read shape).
* Unknown table / column → ``404 NOT_FOUND``.
* On success returns the new ``GeneratorConfig`` as JSON, or — when the
  client carries ``Accept: text/html`` or ``HX-Request: true`` — the
  re-rendered single-row HTMX fragment.
* The workspace spec is replaced *immutably*: the new ``DataSpec`` instance
  differs by identity from the original.
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


def _small_spec() -> DataSpec:
    return DataSpec(
        tables=[
            TableSpec(
                table_name="users",
                row_count=100,
                columns={
                    "id": GeneratorConfig(provider="builtin.sequence", unique=True),
                    "email": GeneratorConfig(provider="mimesis.email"),
                },
            ),
            TableSpec(
                table_name="orders",
                row_count=200,
                columns={
                    "id": GeneratorConfig(provider="builtin.sequence", unique=True),
                    "user_id": GeneratorConfig(provider="numpy.integer"),
                },
            ),
        ],
        schema_hash="deadbeef",
    )


def _seed(app: FastAPI) -> None:
    app.state.workspace.set_schema(_small_schema())
    app.state.workspace.set_spec(_small_spec())


# ── 200 happy path (JSON) ─────────────────────────────────────────────


def test_put_column_updates_spec_and_returns_new_config(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    body = {"provider": "mimesis.first_name", "nullable_rate": 0.25}
    response = client.put("/api/spec/tables/users/columns/email", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "mimesis.first_name"
    assert payload["nullable_rate"] == 0.25
    # Workspace reflects the change.
    spec = app.state.workspace.get_spec()
    users = spec.get_table_spec("users")
    assert users.columns["email"].provider == "mimesis.first_name"
    assert users.columns["email"].nullable_rate == 0.25


def test_put_column_replaces_spec_immutably(tmp_path: Path) -> None:
    """The workspace must end up holding a *new* ``DataSpec`` instance."""
    app = _make_app(tmp_path)
    _seed(app)
    original = app.state.workspace.get_spec()
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/columns/email",
        json={"provider": "mimesis.first_name"},
    )
    assert response.status_code == 200

    after = app.state.workspace.get_spec()
    assert after is not original
    # Original spec instance is untouched (frozen invariants).
    assert original.get_table_spec("users").columns["email"].provider == "mimesis.email"


def test_put_column_preserves_sibling_columns(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    client.put(
        "/api/spec/tables/users/columns/email",
        json={"provider": "mimesis.first_name"},
    )
    spec = app.state.workspace.get_spec()
    users = spec.get_table_spec("users")
    # Sibling column on the same table is untouched.
    assert users.columns["id"].provider == "builtin.sequence"
    assert users.columns["id"].unique is True


# ── 422 validation ────────────────────────────────────────────────────


def test_put_column_rejects_invalid_nullable_rate(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/columns/email",
        json={"provider": "mimesis.email", "nullable_rate": 1.5},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    # At least one entry should reference the ``nullable_rate`` field.
    fields = [tuple(item["loc"]) for item in detail]
    assert any("nullable_rate" in loc for loc in fields)


def test_put_column_rejects_extra_field(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/columns/email",
        json={"provider": "mimesis.email", "not_a_real_field": "boom"},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("not_a_real_field" in tuple(item["loc"]) for item in detail)


def test_put_column_rejects_missing_provider(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/columns/email",
        json={"method": "first_name"},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("provider" in tuple(item["loc"]) for item in detail)


def test_put_column_rejects_non_object_body(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/columns/email",
        json=[1, 2, 3],
    )

    assert response.status_code == 422


# ── 409 constraint guard ──────────────────────────────────────────────


def test_put_column_rejects_pk_dropping_unique(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/columns/id",
        json={"provider": "mimesis.first_name", "unique": False},
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "CONSTRAINT_VIOLATION"
    assert "users" in detail["message"]
    assert detail["table"] == "users"
    assert detail["column"] == "id"


# ── 409 no schema ─────────────────────────────────────────────────────


def test_put_column_no_schema_returns_409_no_schema(tmp_path: Path) -> None:
    """Editing without a loaded schema is the same envelope as the GET read."""
    app = _make_app(tmp_path)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/columns/email",
        json={"provider": "mimesis.email"},
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "NO_SCHEMA"


# ── 404 unknown table / column ────────────────────────────────────────


def test_put_column_unknown_table_returns_404(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/ghosts/columns/id",
        json={"provider": "mimesis.email"},
    )

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["code"] == "NOT_FOUND"
    assert "ghosts" in detail["message"]


def test_put_column_unknown_column_returns_404(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/columns/ghost_col",
        json={"provider": "mimesis.email"},
    )

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["code"] == "NOT_FOUND"
    assert "ghost_col" in detail["message"]


# ── HTML fragment branch ──────────────────────────────────────────────


def test_put_column_html_fragment_via_accept_header(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/columns/email",
        json={"provider": "mimesis.first_name"},
        headers={"Accept": "text/html"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert 'data-column="users.email"' in body
    assert 'data-provider="mimesis.first_name"' in body
    assert 'type="button"' in body


def test_put_column_html_fragment_via_hx_request_header(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/columns/email",
        json={"provider": "mimesis.first_name", "method": "first_name"},
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert 'data-column="users.email"' in body
    assert 'data-method="first_name"' in body
    assert 'data-provider="mimesis.first_name"' in body


# ── coverage corners ──────────────────────────────────────────────────


def test_put_column_rejects_non_json_body(tmp_path: Path) -> None:
    """Raw body that isn't JSON yields a 422 with a friendly note."""
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/columns/email",
        content=b"not json at all { ][",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert any("body" in tuple(item["loc"]) for item in detail)


def test_put_column_builds_spec_when_none_cached(tmp_path: Path) -> None:
    """A PUT before any GET /api/spec lazily builds the spec, then updates it."""
    app = _make_app(tmp_path)
    workspace = app.state.workspace
    workspace.set_schema(_small_schema())
    # Sanity: no spec cached.
    assert workspace.get_spec() is None
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/columns/email",
        json={"provider": "mimesis.first_name"},
    )

    assert response.status_code == 200
    spec = workspace.get_spec()
    assert spec is not None
    users = spec.get_table_spec("users")
    assert users is not None
    assert users.columns["email"].provider == "mimesis.first_name"
