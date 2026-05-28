"""Route-level integration tests for spec-cache persistence (S-122).

The two write endpoints (S-119 column PUT, S-121 table row_count PUT) must
persist the updated ``DataSpec`` to the disk-backed
:class:`~dbsprout.spec.cache.SpecCache` keyed by ``schema.schema_hash()``, and
both schema-load endpoints (S-112 ``POST /api/connect``, S-113
``POST /api/schema/load``) must hydrate the workspace spec from that cache
when a matching entry exists.

These tests inject a ``SpecCache`` over a tmp directory onto
``app.state.workspace`` so they neither touch nor depend on the project's
``.dbsprout/cache``.
"""

from __future__ import annotations

import io
import sqlite3
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
from dbsprout.spec.cache import SpecCache
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
        columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True)],
        primary_key=["id"],
    )
    return DatabaseSchema(tables=[users, orders])


def _temp_sqlite(tmp_path: Path) -> str:
    db_path = tmp_path / "live.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT)")
        conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    return f"sqlite:///{db_path}"


def _wire_cache(app: FastAPI, cache_dir: Path) -> SpecCache:
    """Inject a fresh, tmp-rooted ``SpecCache`` onto the workspace."""
    cache = SpecCache(cache_dir=cache_dir)
    app.state.workspace.set_spec_cache(cache)
    return cache


# ── PUT endpoints → cache write ───────────────────────────────────────


def test_put_table_row_count_persists_spec_to_cache(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    schema = _small_schema()
    app.state.workspace.set_schema(schema)
    cache = _wire_cache(app, tmp_path / "cache")
    try:
        client = TestClient(app)
        client.get("/api/spec")  # build heuristic spec

        resp = client.put("/api/spec/tables/users", json={"row_count": 1234})
        assert resp.status_code == 200, resp.text

        cached = cache.get(schema.schema_hash())
        assert cached is not None
        users = cached.get_table_spec("users")
        assert users is not None
        assert users.row_count == 1234
    finally:
        cache.close()


def test_put_column_persists_spec_to_cache(tmp_path: Path) -> None:
    app = _make_app(tmp_path / "state.db")
    schema = _small_schema()
    app.state.workspace.set_schema(schema)
    cache = _wire_cache(app, tmp_path / "cache")
    try:
        client = TestClient(app)
        client.get("/api/spec")

        resp = client.put(
            "/api/spec/tables/users/columns/email",
            json={"provider": "mimesis", "method": "email"},
        )
        assert resp.status_code == 200, resp.text

        cached = cache.get(schema.schema_hash())
        assert cached is not None
        users = cached.get_table_spec("users")
        assert users is not None
        assert users.columns["email"].provider == "mimesis"
        assert users.columns["email"].method == "email"
    finally:
        cache.close()


def test_put_failure_does_not_persist(tmp_path: Path) -> None:
    """A 422-rejected edit must NOT write to the cache."""
    app = _make_app(tmp_path / "state.db")
    schema = _small_schema()
    app.state.workspace.set_schema(schema)
    cache = _wire_cache(app, tmp_path / "cache")
    try:
        client = TestClient(app)
        client.get("/api/spec")

        # Bad payload — out-of-bounds row_count → 422 from the row_count router.
        resp = client.put("/api/spec/tables/users", json={"row_count": 0})
        assert resp.status_code == 422

        assert cache.get(schema.schema_hash()) is None
    finally:
        cache.close()


# ── schema (re)load → cache hydration ─────────────────────────────────


def test_schema_load_hydrates_spec_from_cache_when_present(tmp_path: Path) -> None:
    """Pre-populate the cache → POST /api/schema/load → workspace.spec is the cached one."""
    app = _make_app(tmp_path / "state.db")
    cache = _wire_cache(app, tmp_path / "cache")
    try:
        schema = _small_schema()
        cached_spec = DataSpec(
            tables=[
                TableSpec(
                    table_name="users",
                    row_count=9999,
                    columns={"id": GeneratorConfig(provider="seq", method="int")},
                ),
                TableSpec(
                    table_name="orders",
                    row_count=42,
                    columns={"id": GeneratorConfig(provider="seq", method="int")},
                ),
            ],
            global_seed=123,
        )
        cache.put(schema.schema_hash(), cached_spec)

        # Upload a DDL that parses to the same schema (matching schema_hash).
        ddl = (
            "CREATE TABLE users (id INTEGER PRIMARY KEY, email VARCHAR(255));"
            "CREATE TABLE orders (id INTEGER PRIMARY KEY);"
        )
        client = TestClient(app)
        resp = client.post(
            "/api/schema/load",
            files={"file": ("schema.sql", io.BytesIO(ddl.encode()), "text/plain")},
        )
        assert resp.status_code == 200, resp.text

        ws_schema = app.state.workspace.get_schema()
        assert ws_schema is not None

        # Now: did hydration set the workspace spec from the cache?
        if ws_schema.schema_hash() == schema.schema_hash():
            ws_spec = app.state.workspace.get_spec()
            assert ws_spec is not None
            assert ws_spec.global_seed == 123
            users = ws_spec.get_table_spec("users")
            assert users is not None
            assert users.row_count == 9999
        else:
            # Parsed schema_hash differs from the manually-built one → register
            # a fresh cache entry against the parsed schema and re-load.
            cache.put(ws_schema.schema_hash(), cached_spec)
            resp = client.post(
                "/api/schema/load",
                files={"file": ("schema.sql", io.BytesIO(ddl.encode()), "text/plain")},
            )
            assert resp.status_code == 200
            ws_spec = app.state.workspace.get_spec()
            assert ws_spec is not None
            assert ws_spec.global_seed == 123
    finally:
        cache.close()


def test_connect_hydrates_spec_from_cache_when_present(tmp_path: Path) -> None:
    """Connect to a live SQLite DB → workspace hydrates from cache if hash matches."""
    app = _make_app(tmp_path / "state.db")
    cache = _wire_cache(app, tmp_path / "cache")
    try:
        url = _temp_sqlite(tmp_path)
        client = TestClient(app)

        # First connect to learn the live schema's hash.
        resp = client.post("/api/connect", json={"url": url})
        assert resp.status_code == 200, resp.text
        live_schema = app.state.workspace.get_schema()
        assert live_schema is not None

        # Seed the cache against the live hash, then reset workspace + reconnect.
        cached_spec = DataSpec(
            tables=[
                TableSpec(
                    table_name="users",
                    row_count=4242,
                    columns={"id": GeneratorConfig(provider="seq", method="int")},
                ),
                TableSpec(
                    table_name="orders",
                    row_count=11,
                    columns={"id": GeneratorConfig(provider="seq", method="int")},
                ),
            ],
            global_seed=77,
        )
        cache.put(live_schema.schema_hash(), cached_spec)
        app.state.workspace.reset()
        app.state.workspace.set_spec_cache(cache)

        resp = client.post("/api/connect", json={"url": url})
        assert resp.status_code == 200, resp.text

        ws_spec = app.state.workspace.get_spec()
        assert ws_spec is not None
        assert ws_spec.global_seed == 77
        users = ws_spec.get_table_spec("users")
        assert users is not None
        assert users.row_count == 4242
    finally:
        cache.close()


def test_connect_no_hydration_on_cache_miss(tmp_path: Path) -> None:
    """An empty cache must leave workspace.spec None — lazy heuristic still applies."""
    app = _make_app(tmp_path / "state.db")
    cache = _wire_cache(app, tmp_path / "cache")
    try:
        url = _temp_sqlite(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/connect", json={"url": url})
        assert resp.status_code == 200

        assert app.state.workspace.get_spec() is None  # no hit → lazy build later
    finally:
        cache.close()


def test_different_schema_hash_does_not_hydrate(tmp_path: Path) -> None:
    """Cache entry under hash A must NOT bleed into a load whose hash is B."""
    app = _make_app(tmp_path / "state.db")
    cache = _wire_cache(app, tmp_path / "cache")
    try:
        # Plant a spec under a synthetic hash unrelated to the real schema.
        unrelated_spec = DataSpec(
            tables=[
                TableSpec(
                    table_name="users",
                    row_count=999,
                    columns={"id": GeneratorConfig(provider="seq", method="int")},
                )
            ],
            global_seed=1,
        )
        cache.put("not-a-real-schema-hash", unrelated_spec)

        url = _temp_sqlite(tmp_path)
        client = TestClient(app)
        resp = client.post("/api/connect", json={"url": url})
        assert resp.status_code == 200

        # Workspace spec must remain unhydrated — the live hash is not the planted one.
        assert app.state.workspace.get_spec() is None
    finally:
        cache.close()


# ── persistence survives a fresh workspace ─────────────────────────────


def test_persisted_spec_survives_workspace_reset(tmp_path: Path) -> None:
    """End-to-end: edit → reset workspace → reconnect → cached spec returns."""
    app = _make_app(tmp_path / "state.db")
    cache = _wire_cache(app, tmp_path / "cache")
    try:
        url = _temp_sqlite(tmp_path)
        client = TestClient(app)

        # Initial connect + edit
        client.post("/api/connect", json={"url": url})
        client.get("/api/spec")
        resp = client.put("/api/spec/tables/users", json={"row_count": 5555})
        assert resp.status_code == 200

        # Simulate "restart": reset workspace, keep cache.
        app.state.workspace.reset()
        app.state.workspace.set_spec_cache(cache)

        # Reconnect → hydration fires.
        client.post("/api/connect", json={"url": url})

        ws_spec = app.state.workspace.get_spec()
        assert ws_spec is not None
        users = ws_spec.get_table_spec("users")
        assert users is not None
        assert users.row_count == 5555
    finally:
        cache.close()
