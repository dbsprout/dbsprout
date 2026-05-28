"""PUT /api/spec/tables/{table} — row_count edit endpoint tests (S-121).

Mirrors ``test_spec_view.py`` (same fixtures + small schema). Locks down:

* JSON happy path returns the new value + the workspace spec is immutably
  updated.
* HTML/HTMX branch returns ``text/html`` with the re-rendered header partial
  carrying the new row_count and the ``data-row-count`` swap target.
* Input validation:
  * ``row_count < 1`` → 422,
  * ``row_count > _DEFAULT_UPPER_BOUND`` → 422,
  * a config-supplied custom bound (smaller than default) is honoured,
  * missing / non-int body → 422.
* Workspace-shape errors:
  * no schema loaded → 409 ``NO_SCHEMA``,
  * unknown table name → 404 ``UNKNOWN_TABLE``.
* Idempotency: same value PUT twice succeeds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.config.models import DBSproutConfig, GenerationConfig
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
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
        columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True)],
        primary_key=["id"],
    )
    return DatabaseSchema(tables=[users, orders])


def _seed_workspace(app: FastAPI, schema: DatabaseSchema) -> None:
    app.state.workspace.set_schema(schema)


# ── JSON happy path ────────────────────────────────────────────────────


def test_put_row_count_returns_new_value_json(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)
    # Build & cache the heuristic spec via GET first.
    client.get("/api/spec")

    response = client.put("/api/spec/tables/users", json={"row_count": 2500})

    assert response.status_code == 200
    body = response.json()
    assert body == {"table_name": "users", "row_count": 2500}


def test_put_row_count_updates_workspace_spec_immutably(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)
    client.get("/api/spec")
    original_spec = app.state.workspace.get_spec()

    client.put("/api/spec/tables/users", json={"row_count": 777})

    updated_spec = app.state.workspace.get_spec()
    assert updated_spec is not None
    # Immutability: a fresh DataSpec instance, not an in-place mutation.
    assert updated_spec is not original_spec
    users = updated_spec.get_table_spec("users")
    assert users is not None
    assert users.row_count == 777


def test_put_row_count_preserves_other_tables(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)
    client.get("/api/spec")
    original = app.state.workspace.get_spec()
    assert original is not None
    orders_before = original.get_table_spec("orders")
    assert orders_before is not None

    client.put("/api/spec/tables/users", json={"row_count": 333})

    spec_after = app.state.workspace.get_spec()
    assert spec_after is not None
    orders_after = spec_after.get_table_spec("orders")
    assert orders_after is not None
    assert orders_after.row_count == orders_before.row_count


# ── HTMX / HTML fragment branch ──────────────────────────────────────────


def test_put_row_count_returns_header_fragment_for_htmx(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)
    client.get("/api/spec")

    response = client.put(
        "/api/spec/tables/users",
        json={"row_count": 444},
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "data-row-count" in body
    assert "444" in body


def test_put_row_count_returns_header_fragment_for_accept_html(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)
    client.get("/api/spec")

    response = client.put(
        "/api/spec/tables/users",
        json={"row_count": 88},
        headers={"Accept": "text/html"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "data-row-count" in body
    assert "88" in body


# ── input validation: 422 envelope ────────────────────────────────────


def test_put_row_count_zero_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)
    client.get("/api/spec")

    response = client.put("/api/spec/tables/users", json={"row_count": 0})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "INVALID_ROW_COUNT"
    assert "row_count" in detail["message"].lower()


def test_put_row_count_negative_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)
    client.get("/api/spec")

    response = client.put("/api/spec/tables/users", json={"row_count": -1})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "INVALID_ROW_COUNT"


def test_put_row_count_above_default_bound_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)
    client.get("/api/spec")

    response = client.put(
        "/api/spec/tables/users",
        json={"row_count": 10_000_001},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "INVALID_ROW_COUNT"
    assert "10000000" in detail["message"] or "10_000_000" in detail["message"]


def test_put_row_count_at_default_bound_succeeds(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)
    client.get("/api/spec")

    response = client.put(
        "/api/spec/tables/users",
        json={"row_count": 10_000_000},
    )

    assert response.status_code == 200
    assert response.json()["row_count"] == 10_000_000


def test_put_row_count_honours_custom_upper_bound_from_config(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    # Inject a custom config with a much smaller bound on the app state.
    app.state.config = DBSproutConfig(
        generation=GenerationConfig(max_rows_per_table=500),
    )
    client = TestClient(app)
    client.get("/api/spec")

    over_bound = client.put("/api/spec/tables/users", json={"row_count": 501})
    at_bound = client.put("/api/spec/tables/users", json={"row_count": 500})

    assert over_bound.status_code == 422
    assert over_bound.json()["detail"]["code"] == "INVALID_ROW_COUNT"
    assert "500" in over_bound.json()["detail"]["message"]
    assert at_bound.status_code == 200
    assert at_bound.json()["row_count"] == 500


def test_put_row_count_missing_body_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)
    client.get("/api/spec")

    response = client.put("/api/spec/tables/users", json={})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_ROW_COUNT"


def test_put_row_count_non_int_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)
    client.get("/api/spec")

    response = client.put(
        "/api/spec/tables/users",
        json={"row_count": "lots"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_ROW_COUNT"


def test_put_row_count_float_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)
    client.get("/api/spec")

    response = client.put(
        "/api/spec/tables/users",
        json={"row_count": 1.5},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_ROW_COUNT"


# ── workspace-shape errors ───────────────────────────────────────────


def test_put_row_count_no_schema_returns_409(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)

    response = client.put("/api/spec/tables/users", json={"row_count": 100})

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "NO_SCHEMA"


def test_put_row_count_unknown_table_returns_404(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)
    client.get("/api/spec")

    response = client.put(
        "/api/spec/tables/ghosts",
        json={"row_count": 100},
    )

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["code"] == "UNKNOWN_TABLE"
    assert "ghosts" in detail["message"]


# ── idempotency ─────────────────────────────────────────────────────


def test_put_row_count_non_dict_body_returns_422(tmp_path: Path) -> None:
    """A JSON *array* (or any non-object) body is rejected with 422."""
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)
    client.get("/api/spec")

    response = client.put("/api/spec/tables/users", json=[123])

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_ROW_COUNT"


def test_put_row_count_malformed_json_returns_422(tmp_path: Path) -> None:
    """A body that is not valid JSON falls back to the 422 envelope."""
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)
    client.get("/api/spec")

    response = client.put(
        "/api/spec/tables/users",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_ROW_COUNT"


def test_put_row_count_config_without_generation_uses_default_bound(
    tmp_path: Path,
) -> None:
    """``app.state.config`` set to a value WITHOUT ``generation`` still works.

    Defence-in-depth: the resolver tolerates any object on ``state.config``
    (it may be set by a future S-122 cache layer); a missing ``generation`` or
    ``max_rows_per_table`` simply falls back to the default bound.
    """
    from types import SimpleNamespace  # noqa: PLC0415

    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    app.state.config = SimpleNamespace()  # no ``generation`` attribute
    client = TestClient(app)
    client.get("/api/spec")

    # Default bound (10M) still in effect: a value at the default ceiling works.
    response = client.put(
        "/api/spec/tables/users",
        json={"row_count": 10_000_000},
    )
    assert response.status_code == 200


def test_put_row_count_config_with_generation_but_no_bound_uses_default(
    tmp_path: Path,
) -> None:
    """``config.generation.max_rows_per_table = None`` → default bound applies."""
    from types import SimpleNamespace  # noqa: PLC0415

    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    app.state.config = SimpleNamespace(
        generation=SimpleNamespace(max_rows_per_table=None),
    )
    client = TestClient(app)
    client.get("/api/spec")

    response = client.put(
        "/api/spec/tables/users",
        json={"row_count": 10_000_000},
    )
    assert response.status_code == 200


def test_put_row_count_is_idempotent_on_same_value(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, _small_schema())
    client = TestClient(app)
    client.get("/api/spec")

    first = client.put("/api/spec/tables/users", json={"row_count": 123})
    second = client.put("/api/spec/tables/users", json={"row_count": 123})

    assert first.status_code == 200
    assert second.status_code == 200
    spec = app.state.workspace.get_spec()
    assert spec is not None
    users = spec.get_table_spec("users")
    assert users is not None
    assert users.row_count == 123
