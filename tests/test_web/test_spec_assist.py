"""POST /api/spec/assist — AI spec-assist route tests (P2b-3).

The route re-exposes the LLM-as-architect capability (deleted with the legacy
wizard in P1c-5) as a JSON endpoint: it loads the schema from the in-memory
:class:`~dbsprout.web.workspace.Workspace` (``app.state.workspace``), calls the
chosen provider's ``generate_spec(schema) -> DataSpec``, stores the proposed
spec on the workspace (so ``GET /api/spec`` + the grid reflect it), and returns
a summary.

The web stack lives in the optional ``[web]`` extra, so this module guards with
``pytest.importorskip("fastapi")`` before importing FastAPI symbols. The two
providers live behind the ``[llm]`` / ``[cloud]`` extras (llama-cpp / litellm),
neither of which is installed in CI — so **every test monkeypatches the provider
class** with a stand-in. The route must therefore lazy-import the providers
*inside the handler*; importing the router itself must pull neither extra.

Tests seed ``app.state.workspace`` directly (schema only) rather than going
through ``POST /api/connect`` — the upstream routes have their own tests, and
this keeps each unit small.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

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


# ── helpers ─────────────────────────────────────────────────────────────


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


def _proposed_spec(schema: DatabaseSchema, *, model: str = "fake-llm") -> DataSpec:
    """A distinctive DataSpec a stub provider returns, marked so tests can
    prove the *LLM* spec (not the heuristic fallback) landed on the workspace."""
    return DataSpec(
        tables=[
            TableSpec(
                table_name="users",
                row_count=123,
                columns={
                    "id": GeneratorConfig(provider="builtin", method="autoincrement"),
                    "email": GeneratorConfig(provider="llm.marker", method="email"),
                },
            ),
            TableSpec(
                table_name="orders",
                row_count=45,
                columns={
                    "id": GeneratorConfig(provider="builtin", method="autoincrement"),
                    "user_id": GeneratorConfig(provider="builtin", method="fk"),
                },
            ),
        ],
        model_used=model,
        schema_hash=schema.schema_hash(),
    )


class _StubProvider:
    """Stand-in for Embedded/Cloud providers — records the call, returns a spec."""

    instances: ClassVar[list[_StubProvider]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.calls: list[DatabaseSchema] = []
        _StubProvider.instances.append(self)

    def generate_spec(self, schema: DatabaseSchema) -> DataSpec:
        self.calls.append(schema)
        return _proposed_spec(schema)


@pytest.fixture(autouse=True)
def _reset_stub() -> None:
    _StubProvider.instances.clear()


def _patch_embedded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dbsprout.spec.providers.embedded.EmbeddedProvider",
        _StubProvider,
        raising=True,
    )


def _patch_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dbsprout.spec.providers.cloud.CloudProvider",
        _StubProvider,
        raising=True,
    )


# ── import-only contract ─────────────────────────────────────────────────


def test_router_imports_without_llm_or_cloud_extras() -> None:
    """Importing the router must not pull ``llama-cpp`` / ``litellm`` (CI has
    neither). A plain import is the cheapest proof of the lazy-import contract."""
    import dbsprout.web.routers.spec_assist as mod  # noqa: PLC0415

    assert hasattr(mod, "spec_assist_router")


# ── happy path: embedded default ─────────────────────────────────────────


def test_assist_embedded_default_proposes_spec_onto_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_embedded(monkeypatch)
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())
    client = TestClient(app)

    resp = client.post("/api/spec/assist", json={})

    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "embedded"
    assert body["model_used"] == "fake-llm"
    assert body["tables"] == 2
    assert body["total_columns"] == 4
    # The provider was actually called with the loaded schema.
    assert len(_StubProvider.instances) == 1
    assert len(_StubProvider.instances[0].calls) == 1
    # The proposed (LLM) spec — not the heuristic fallback — is now on the workspace.
    stored = app.state.workspace.get_spec()
    assert stored is not None
    assert stored.model_used == "fake-llm"
    assert stored.tables[0].columns["email"].provider == "llm.marker"


def test_assist_then_get_spec_reflects_proposed_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC: after assist, ``GET /api/spec`` returns the proposed DataSpec."""
    _patch_embedded(monkeypatch)
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())
    client = TestClient(app)

    client.post("/api/spec/assist", json={"provider": "embedded"})
    got = client.get("/api/spec")

    assert got.status_code == 200
    spec = got.json()
    assert spec["model_used"] == "fake-llm"
    assert spec["tables"][0]["row_count"] == 123
    assert spec["tables"][0]["columns"]["email"]["provider"] == "llm.marker"


def test_assist_caches_proposed_spec_by_schema_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC: the proposed spec is cached by ``schema_hash`` (survives a fresh
    Workspace reading the same disk cache)."""
    _patch_embedded(monkeypatch)
    app = _make_app(tmp_path)
    schema = _small_schema()
    app.state.workspace.set_schema(schema)
    client = TestClient(app)

    resp = client.post("/api/spec/assist", json={})
    assert resp.status_code == 200

    # A brand-new workspace hydrating from the *same* disk cache finds the spec.
    from dbsprout.web.workspace import Workspace  # noqa: PLC0415

    fresh = Workspace()
    fresh.set_spec_cache(app.state.workspace._get_spec_cache())
    assert fresh.hydrate_from_cache(schema.schema_hash()) is True
    assert fresh.get_spec() is not None
    assert fresh.get_spec().model_used == "fake-llm"  # type: ignore[union-attr]


# ── explicit cloud provider ──────────────────────────────────────────────


def test_assist_explicit_cloud_calls_cloud_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_cloud(monkeypatch)
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())
    client = TestClient(app)

    resp = client.post("/api/spec/assist", json={"provider": "cloud"})

    assert resp.status_code == 200
    assert resp.json()["provider"] == "cloud"
    assert len(_StubProvider.instances) == 1


# ── P4-11: cloud key-entry UX (model + env-var reference) ────────────────────
#
# The cloud assist call may carry a non-secret ``model`` string and an
# ``api_key_env`` *name* (e.g. ``OPENAI_API_KEY``). The raw key never crosses the
# wire — litellm reads it from the server's own process environment. The route
# guards on env-var presence so a missing key degrades to the typed 503 envelope
# (naming the var) instead of a deep provider failure, and forwards ``model`` to
# ``CloudProvider(model=...)``.


def test_assist_cloud_forwards_model_to_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``model`` on the cloud request is forwarded to ``CloudProvider(model=...)``."""
    _patch_cloud(monkeypatch)
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())
    client = TestClient(app)

    resp = client.post(
        "/api/spec/assist",
        json={"provider": "cloud", "model": "gpt-4o-mini"},
    )

    assert resp.status_code == 200
    assert len(_StubProvider.instances) == 1
    assert _StubProvider.instances[0].kwargs.get("model") == "gpt-4o-mini"


def test_assist_cloud_missing_env_var_returns_typed_503_naming_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``api_key_env`` naming an *unset* var → typed 503 that names the var, and
    the provider is never even constructed (so no real API call is attempted)."""
    _patch_cloud(monkeypatch)
    monkeypatch.delenv("DEFINITELY_UNSET_KEY_P411", raising=False)
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())
    client = TestClient(app)

    resp = client.post(
        "/api/spec/assist",
        json={"provider": "cloud", "api_key_env": "DEFINITELY_UNSET_KEY_P411"},
    )

    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["code"] == "LLM_UNAVAILABLE"
    # The message must name the missing variable so the UX can act on it.
    assert "DEFINITELY_UNSET_KEY_P411" in detail["message"]
    # Provider never constructed; no spec committed on the failure path.
    assert len(_StubProvider.instances) == 0
    assert app.state.workspace.get_spec() is None


def test_assist_cloud_present_env_var_constructs_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``api_key_env`` naming a *present* var → the provider is constructed and the
    proposal succeeds (litellm would read the key from the same environment)."""
    _patch_cloud(monkeypatch)
    monkeypatch.setenv("PRESENT_KEY_P411", "sk-not-a-real-secret")
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())
    client = TestClient(app)

    resp = client.post(
        "/api/spec/assist",
        json={"provider": "cloud", "api_key_env": "PRESENT_KEY_P411", "model": "gpt-4o"},
    )

    assert resp.status_code == 200
    assert resp.json()["provider"] == "cloud"
    assert len(_StubProvider.instances) == 1
    assert _StubProvider.instances[0].kwargs.get("model") == "gpt-4o"


def test_assist_embedded_ignores_model_and_env_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The embedded (offline) path ignores ``model`` / ``api_key_env`` — the
    env-var guard is cloud-only, and the embedded provider takes no model arg."""
    _patch_embedded(monkeypatch)
    monkeypatch.delenv("SOME_UNSET_KEY_P411", raising=False)
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())
    client = TestClient(app)

    resp = client.post(
        "/api/spec/assist",
        json={
            "provider": "embedded",
            "model": "ignored",
            "api_key_env": "SOME_UNSET_KEY_P411",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["provider"] == "embedded"
    assert len(_StubProvider.instances) == 1


def test_assist_cloud_no_env_field_unchanged_behaviour(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting ``api_key_env`` keeps the legacy behaviour: the provider is
    constructed and litellm itself raises if no key is configured (folded into
    the typed 503 by the existing broad-except path)."""
    _patch_cloud(monkeypatch)
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())
    client = TestClient(app)

    resp = client.post("/api/spec/assist", json={"provider": "cloud", "model": "gpt-4o"})

    assert resp.status_code == 200
    assert len(_StubProvider.instances) == 1


# ── error paths ──────────────────────────────────────────────────────────


def test_assist_no_schema_returns_409_no_schema(tmp_path: Path) -> None:
    app = _make_app(tmp_path)  # no schema seeded
    client = TestClient(app)

    resp = client.post("/api/spec/assist", json={})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "NO_SCHEMA"


def test_assist_provider_import_error_returns_typed_503_never_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing ``[llm]`` extra → ``EmbeddedProvider`` raises ImportError on
    construction or call → typed LLM_UNAVAILABLE envelope, NEVER a 500. The
    existing workspace spec (if any) is left untouched."""

    class _Boom:
        def __init__(self, *a: object, **k: object) -> None:
            msg = "llama-cpp-python is required for embedded LLM inference."
            raise ImportError(msg)

    monkeypatch.setattr("dbsprout.spec.providers.embedded.EmbeddedProvider", _Boom, raising=True)
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())
    client = TestClient(app)

    resp = client.post("/api/spec/assist", json={"provider": "embedded"})

    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["code"] == "LLM_UNAVAILABLE"
    assert "message" in detail
    # No spec was committed on the failure path.
    assert app.state.workspace.get_spec() is None


def test_assist_provider_runtime_error_returns_typed_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runtime failure during ``generate_spec`` (e.g. no GGUF model on disk,
    or a missing cloud API key) is also folded into the typed envelope."""

    class _RuntimeBoom:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def generate_spec(self, schema: DatabaseSchema) -> DataSpec:
            msg = "no model file found and no API key configured"
            raise RuntimeError(msg)

    monkeypatch.setattr("dbsprout.spec.providers.cloud.CloudProvider", _RuntimeBoom, raising=True)
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())
    client = TestClient(app)

    resp = client.post("/api/spec/assist", json={"provider": "cloud"})

    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "LLM_UNAVAILABLE"


def test_assist_provider_arbitrary_exception_returns_typed_503_never_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing / invalid cloud API key surfaces as litellm's own
    ``AuthenticationError`` — a plain ``Exception`` subclass, none of the stdlib
    types. The route must still fold it into the typed 503 envelope (the AC
    mandates this path NEVER returns a 500)."""

    class _AuthError(Exception):
        """Stand-in for litellm.exceptions.AuthenticationError (not a stdlib type)."""

    class _AuthBoom:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def generate_spec(self, schema: DatabaseSchema) -> DataSpec:
            raise _AuthError("AuthenticationError: no API key provided")

    monkeypatch.setattr("dbsprout.spec.providers.cloud.CloudProvider", _AuthBoom, raising=True)
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())
    client = TestClient(app)

    resp = client.post("/api/spec/assist", json={"provider": "cloud"})

    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "LLM_UNAVAILABLE"
    assert app.state.workspace.get_spec() is None


def test_assist_unknown_provider_value_returns_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())
    client = TestClient(app)

    resp = client.post("/api/spec/assist", json={"provider": "wizardry"})

    assert resp.status_code == 422


def test_assist_extra_field_rejected_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())
    client = TestClient(app)

    resp = client.post("/api/spec/assist", json={"provider": "embedded", "bogus": 1})

    assert resp.status_code == 422
