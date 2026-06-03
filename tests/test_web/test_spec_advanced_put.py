"""``PUT /api/spec/tables/{t}/advanced`` — correlations + derived edit (P2b-2).

The endpoint extends the spec router to persist a table's *advanced packs*:

* JSON body ``{"correlations"?: [...], "derived"?: [...]}`` — either key may be
  omitted (partial update); the omitted list is left unchanged.
* ``correlations`` entries are :class:`~dbsprout.spec.models.CorrelationRule`,
  ``derived`` entries are :class:`~dbsprout.spec.models.DerivedColumn`; bad input
  yields ``422`` with field-level Pydantic errors.
* Every referenced column (``CorrelationRule.columns``, ``DerivedColumn.column``
  and ``.depends_on``) must exist on the *loaded schema* table; an unknown
  reference yields ``422 UNKNOWN_COLUMN_REF``. A ``CorrelationRule.lookup_table``
  that is not a real table yields ``422 UNKNOWN_LOOKUP_TABLE``.
* Missing schema → ``409 NO_SCHEMA`` (matches the rest of the spec router).
* Unknown table → ``404 UNKNOWN_TABLE``.
* On success returns ``{table_name, correlations, derived}`` as JSON; the
  workspace spec is replaced *immutably*.
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
from dbsprout.spec.models import (
    CorrelationRule,
    DataSpec,
    DerivedColumn,
    GeneratorConfig,
    TableSpec,
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
            ColumnSchema(name="city", data_type=ColumnType.VARCHAR, max_length=255),
            ColumnSchema(name="state", data_type=ColumnType.VARCHAR, max_length=64),
            ColumnSchema(name="zip", data_type=ColumnType.VARCHAR, max_length=16),
        ],
        primary_key=["id"],
    )
    orders = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name="user_id", data_type=ColumnType.INTEGER),
            ColumnSchema(name="qty", data_type=ColumnType.INTEGER),
            ColumnSchema(name="price", data_type=ColumnType.FLOAT),
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
                    "city": GeneratorConfig(provider="mimesis.city"),
                    "state": GeneratorConfig(provider="mimesis.state"),
                    "zip": GeneratorConfig(provider="mimesis.zip_code"),
                },
            ),
            TableSpec(
                table_name="orders",
                row_count=200,
                columns={
                    "id": GeneratorConfig(provider="builtin.sequence", unique=True),
                    "user_id": GeneratorConfig(provider="numpy.integer"),
                    "qty": GeneratorConfig(provider="numpy.integer"),
                    "price": GeneratorConfig(provider="numpy.float"),
                },
            ),
        ],
        schema_hash="deadbeef",
    )


def _seed(app: FastAPI) -> None:
    app.state.workspace.set_schema(_small_schema())
    app.state.workspace.set_spec(_small_spec())


# ── 200 happy path (JSON) ─────────────────────────────────────────────


def test_put_advanced_sets_correlations(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    body = {
        "correlations": [
            {"columns": ["city", "state", "zip"], "strategy": "lookup"},
        ],
    }
    response = client.put("/api/spec/tables/users/advanced", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["table_name"] == "users"
    assert payload["correlations"][0]["columns"] == ["city", "state", "zip"]
    # Workspace reflects the change.
    users = app.state.workspace.get_spec().get_table_spec("users")
    assert users.correlations == [
        CorrelationRule(columns=["city", "state", "zip"], strategy="lookup"),
    ]


def test_put_advanced_sets_derived(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    body = {
        "derived": [
            {"column": "price", "expression": "qty * 9.99", "depends_on": ["qty"]},
        ],
    }
    response = client.put("/api/spec/tables/orders/advanced", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["derived"][0]["column"] == "price"
    orders = app.state.workspace.get_spec().get_table_spec("orders")
    assert orders.derived == [
        DerivedColumn(column="price", expression="qty * 9.99", depends_on=["qty"]),
    ]


def test_put_advanced_sets_both(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    body = {
        "correlations": [{"columns": ["qty", "price"]}],
        "derived": [
            {"column": "price", "expression": "qty * 2", "depends_on": ["qty"]},
        ],
    }
    response = client.put("/api/spec/tables/orders/advanced", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["correlations"]) == 1
    assert len(payload["derived"]) == 1


def test_put_advanced_replaces_spec_immutably(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    original = app.state.workspace.get_spec()
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/advanced",
        json={"correlations": [{"columns": ["city", "state"]}]},
    )
    assert response.status_code == 200

    after = app.state.workspace.get_spec()
    assert after is not original
    # Original spec instance is untouched (frozen invariants).
    assert original.get_table_spec("users").correlations == []


def test_put_advanced_partial_update_keeps_other_list(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    client.put(
        "/api/spec/tables/orders/advanced",
        json={"derived": [{"column": "price", "expression": "qty*2", "depends_on": ["qty"]}]},
    )
    # Now update only correlations — derived must survive.
    client.put(
        "/api/spec/tables/orders/advanced",
        json={"correlations": [{"columns": ["qty", "price"]}]},
    )

    orders = app.state.workspace.get_spec().get_table_spec("orders")
    assert orders.derived == [
        DerivedColumn(column="price", expression="qty*2", depends_on=["qty"]),
    ]
    assert orders.correlations == [CorrelationRule(columns=["qty", "price"])]


def test_put_advanced_empty_body_is_noop_success(tmp_path: Path) -> None:
    """An empty object leaves both lists untouched and returns them."""
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put("/api/spec/tables/users/advanced", json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload["correlations"] == []
    assert payload["derived"] == []


# ── 409 no schema ─────────────────────────────────────────────────────


def test_put_advanced_no_schema_returns_409(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/advanced",
        json={"correlations": [{"columns": ["city"]}]},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "NO_SCHEMA"


# ── 404 unknown table ─────────────────────────────────────────────────


def test_put_advanced_unknown_table_returns_404(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/ghosts/advanced",
        json={"correlations": []},
    )

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["code"] == "UNKNOWN_TABLE"
    assert "ghosts" in detail["message"]


# ── 422 unknown column references ─────────────────────────────────────


def test_put_advanced_unknown_correlation_column_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/advanced",
        json={"correlations": [{"columns": ["city", "ghost_col"]}]},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "UNKNOWN_COLUMN_REF"
    assert detail["column"] == "ghost_col"
    assert detail["table"] == "users"


def test_put_advanced_unknown_derived_column_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/orders/advanced",
        json={
            "derived": [
                {"column": "ghost_target", "expression": "qty*2", "depends_on": ["qty"]},
            ],
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "UNKNOWN_COLUMN_REF"
    assert detail["column"] == "ghost_target"


def test_put_advanced_unknown_depends_on_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/orders/advanced",
        json={
            "derived": [
                {"column": "price", "expression": "ghost*2", "depends_on": ["ghost"]},
            ],
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "UNKNOWN_COLUMN_REF"
    assert detail["column"] == "ghost"


def test_put_advanced_unknown_lookup_table_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/advanced",
        json={
            "correlations": [
                {"columns": ["city", "state"], "lookup_table": "ghost_table"},
            ],
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "UNKNOWN_LOOKUP_TABLE"
    assert "ghost_table" in detail["message"]


def test_put_advanced_known_lookup_table_succeeds(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/advanced",
        json={
            "correlations": [
                {"columns": ["city", "state"], "lookup_table": "orders"},
            ],
        },
    )
    assert response.status_code == 200


# ── 422 validation (pydantic / malformed) ─────────────────────────────


def test_put_advanced_extra_field_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/advanced",
        json={"not_a_real_field": "boom"},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert any("not_a_real_field" in tuple(item["loc"]) for item in detail)


def test_put_advanced_bad_correlation_shape_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/advanced",
        json={"correlations": [{"strategy": "lookup"}]},  # missing required ``columns``
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list)


def test_put_advanced_non_object_body_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put("/api/spec/tables/users/advanced", json=[1, 2, 3])
    assert response.status_code == 422


def test_put_advanced_malformed_json_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/advanced",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert any("body" in tuple(item["loc"]) for item in detail)


# ── lazy build + JSON-only ────────────────────────────────────────────


def test_put_advanced_builds_spec_when_none_cached(tmp_path: Path) -> None:
    """A PUT before any GET /api/spec lazily builds the spec, then updates it."""
    app = _make_app(tmp_path)
    workspace = app.state.workspace
    workspace.set_schema(_small_schema())
    assert workspace.get_spec() is None
    client = TestClient(app)

    response = client.put(
        "/api/spec/tables/users/advanced",
        json={"correlations": [{"columns": ["city", "state"]}]},
    )

    assert response.status_code == 200
    spec = workspace.get_spec()
    assert spec is not None
    users = spec.get_table_spec("users")
    assert users.correlations == [CorrelationRule(columns=["city", "state"])]


def test_put_advanced_returns_json_even_with_accept_html(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed(app)
    client = TestClient(app)

    for headers in ({"Accept": "text/html"}, {"HX-Request": "true"}):
        response = client.put(
            "/api/spec/tables/users/advanced",
            json={"correlations": [{"columns": ["city"]}]},
            headers=headers,
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
