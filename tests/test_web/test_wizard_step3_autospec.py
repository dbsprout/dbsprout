"""Wizard Step 3 (Configure) auto-spec + LLM opt-in tests (S-145).

When the user lands on Step 3 the wizard now proactively populates a
heuristic ``DataSpec`` on the workspace so the spec grid renders
immediately, no spinner. A separate ``POST /wizard/step/3/llm-spec``
endpoint lets the user opt into the slower LLM path; when the local LLM
provider is unavailable (no GGUF, no ``llama-cpp-python``) it surfaces a
``503 LLM_UNAVAILABLE`` envelope rather than crashing.

The tests pin:

* the GET-side entry hook (heuristic build → cached spec preferred →
  no-schema → no heavy LLM import on the read path),
* the new POST endpoint (no-schema → 409, provider unavailable → 503,
  happy path replaces the workspace spec + persists),
* the template surface (Use-LLM button + HTMX attributes).

Sibling stories edit other regions of the same files (S-144 wraps the
Studio shell button, S-146 adds help-icon macros to the rail) — these
tests stay scoped to the Step 3 body + new POST route to keep wave merge
clean.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)
from dbsprout.spec.models import DataSpec, GeneratorConfig, TableSpec

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


# ── helpers ──────────────────────────────────────────────────────────────


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
    return DatabaseSchema(tables=[users])


def _sentinel_spec(schema_hash: str, *, marker: str = "from_cache") -> DataSpec:
    """Build a tiny DataSpec we can identify by ``model_used`` later."""
    return DataSpec(
        tables=[
            TableSpec(
                table_name="users",
                columns={
                    "id": GeneratorConfig(provider="builtin.default"),
                    "email": GeneratorConfig(provider="builtin.default"),
                },
            )
        ],
        schema_hash=schema_hash,
        model_used=marker,
    )


# ── GET /wizard/step/3 — auto heuristic build ────────────────────────────


def test_get_step3_builds_heuristic_spec_when_absent(tmp_path: Path) -> None:
    """No cached spec + schema present → heuristic spec written to workspace."""
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())
    assert app.state.workspace.get_spec() is None

    response = TestClient(app).get("/wizard/step/3")

    assert response.status_code == 200, response.text
    spec_after = app.state.workspace.get_spec()
    assert spec_after is not None
    # Heuristic build path stamps ``heuristic_fallback`` on ``model_used``.
    assert spec_after.model_used == "heuristic_fallback"


def test_get_step3_prefers_cached_spec_over_fresh_build(tmp_path: Path) -> None:
    """If hydrate-from-cache returned a spec, the entry hook keeps it untouched."""
    app = _make_app(tmp_path)
    schema = _small_schema()
    app.state.workspace.set_schema(schema)
    # Simulate S-122's hydrate-from-cache having already populated spec.
    cached = _sentinel_spec(schema.schema_hash(), marker="from_cache_marker")
    app.state.workspace.set_spec(cached)

    TestClient(app).get("/wizard/step/3")

    spec_after = app.state.workspace.get_spec()
    assert spec_after is cached, "entry hook clobbered an existing cached spec"
    assert spec_after.model_used == "from_cache_marker"


def test_get_step3_with_no_schema_does_not_set_spec(tmp_path: Path) -> None:
    """No schema → no heuristic build → spec stays None, body still renders."""
    app = _make_app(tmp_path)
    assert app.state.workspace.get_schema() is None

    response = TestClient(app).get("/wizard/step/3")

    assert response.status_code == 200
    assert app.state.workspace.get_spec() is None


def test_get_step3_does_not_import_embedded_llm_provider(tmp_path: Path) -> None:
    """Read-path stays import-light: the LLM provider must not load on Step 3 GET."""
    # Drop the LLM provider from sys.modules so we can prove it isn't re-imported.
    sys.modules.pop("dbsprout.spec.providers.embedded", None)
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())

    TestClient(app).get("/wizard/step/3")

    assert "dbsprout.spec.providers.embedded" not in sys.modules, (
        "Step 3 GET pulled in the heavy embedded LLM provider"
    )


def test_get_step3_uses_cached_spec_via_workspace_hydrate(tmp_path: Path) -> None:
    """When a spec is sitting in the disk cache, the entry hook hydrates it."""
    from dbsprout.spec.cache import SpecCache  # noqa: PLC0415

    app = _make_app(tmp_path)
    schema = _small_schema()
    app.state.workspace.set_schema(schema)

    # Wire a tmp-rooted cache, prime it with a spec we can identify, then
    # confirm the entry hook hydrates rather than rebuilding.
    cache = SpecCache(cache_dir=tmp_path / "spec_cache")
    app.state.workspace.set_spec_cache(cache)
    primed = _sentinel_spec(schema.schema_hash(), marker="primed_in_cache")
    cache.put(schema.schema_hash(), primed)

    TestClient(app).get("/wizard/step/3")

    spec_after = app.state.workspace.get_spec()
    assert spec_after is not None
    assert spec_after.model_used == "primed_in_cache"


# ── template surface — Use-LLM button ────────────────────────────────────


def test_get_step3_template_renders_use_llm_button(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())

    body = TestClient(app).get("/wizard/step/3").text

    # Button copy + the HTMX wiring that targets the new endpoint.
    assert "Use LLM" in body
    assert "/wizard/step/3/llm-spec" in body
    assert "hx-post" in body


# ── POST /wizard/step/3/llm-spec — opt-in LLM path ───────────────────────


def test_post_llm_spec_returns_409_when_no_schema(tmp_path: Path) -> None:
    app = _make_app(tmp_path)

    response = TestClient(app).post("/wizard/step/3/llm-spec")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "NO_SCHEMA"


def test_post_llm_spec_returns_503_when_provider_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ImportError`` from the embedded LLM provider → 503 LLM_UNAVAILABLE."""
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())

    # Force the lazy ``EmbeddedProvider`` constructor to fail as if llama-cpp /
    # the GGUF model were absent. The wizard route catches and translates.
    from dbsprout.spec.providers import embedded as embedded_mod  # noqa: PLC0415

    def _boom(self: Any, *args: object, **kwargs: object) -> None:
        msg = "llama-cpp-python not installed"
        raise ImportError(msg)

    monkeypatch.setattr(embedded_mod.EmbeddedProvider, "__init__", _boom)

    response = TestClient(app).post("/wizard/step/3/llm-spec")

    assert response.status_code == 503, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "LLM_UNAVAILABLE"
    assert "message" in detail
    # Heuristic spec is still in place (or absent — either way no crash).


def test_post_llm_spec_returns_503_when_provider_raises_runtime_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider construction raising RuntimeError → 503 LLM_UNAVAILABLE."""
    app = _make_app(tmp_path)
    app.state.workspace.set_schema(_small_schema())

    from dbsprout.spec.providers import embedded as embedded_mod  # noqa: PLC0415

    def _boom(self: Any, *args: object, **kwargs: object) -> None:
        msg = "no GGUF model found"
        raise RuntimeError(msg)

    monkeypatch.setattr(embedded_mod.EmbeddedProvider, "__init__", _boom)

    response = TestClient(app).post("/wizard/step/3/llm-spec")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "LLM_UNAVAILABLE"


def test_post_llm_spec_happy_path_replaces_workspace_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful LLM analyze → workspace spec swapped, response 200."""
    from dbsprout.spec.cache import SpecCache  # noqa: PLC0415

    app = _make_app(tmp_path)
    schema = _small_schema()
    app.state.workspace.set_schema(schema)
    # Pin a tmp-rooted cache so ``persist_spec`` doesn't poison the project
    # default ``.dbsprout/cache`` for sibling tests.
    app.state.workspace.set_spec_cache(SpecCache(cache_dir=tmp_path / "spec_cache"))
    llm_spec = _sentinel_spec(schema.schema_hash(), marker="from_llm")

    # Replace SpecAnalyzer with a tiny double so we don't load a real GGUF.
    class _FakeAnalyzer:
        def __init__(self, provider: object) -> None:
            self._provider = provider

        def analyze(self, _schema: DatabaseSchema) -> DataSpec:
            return llm_spec

        def get_last_usage(self) -> None:
            return None

    class _FakeProvider:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return

    from dbsprout.spec import analyzer as analyzer_mod  # noqa: PLC0415
    from dbsprout.spec.providers import embedded as embedded_mod  # noqa: PLC0415

    monkeypatch.setattr(analyzer_mod, "SpecAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr(embedded_mod, "EmbeddedProvider", _FakeProvider)

    response = TestClient(app).post(
        "/wizard/step/3/llm-spec",
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200, response.text
    stored = app.state.workspace.get_spec()
    assert stored is not None
    assert stored.model_used == "from_llm"
    # HTMX body re-renders the Step 3 fragment.
    assert "wizard-spec-grid-slot" in response.text


def test_post_llm_spec_json_response_when_not_htmx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-HTMX caller gets a small JSON ack instead of an HTML re-render."""
    from dbsprout.spec.cache import SpecCache  # noqa: PLC0415

    app = _make_app(tmp_path)
    schema = _small_schema()
    app.state.workspace.set_schema(schema)
    app.state.workspace.set_spec_cache(SpecCache(cache_dir=tmp_path / "spec_cache"))
    llm_spec = _sentinel_spec(schema.schema_hash(), marker="from_llm_json")

    class _FakeAnalyzer:
        def __init__(self, provider: object) -> None:
            return

        def analyze(self, _schema: DatabaseSchema) -> DataSpec:
            return llm_spec

        def get_last_usage(self) -> None:
            return None

    class _FakeProvider:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return

    from dbsprout.spec import analyzer as analyzer_mod  # noqa: PLC0415
    from dbsprout.spec.providers import embedded as embedded_mod  # noqa: PLC0415

    monkeypatch.setattr(analyzer_mod, "SpecAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr(embedded_mod, "EmbeddedProvider", _FakeProvider)

    response = TestClient(app).post("/wizard/step/3/llm-spec")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is True
    assert payload["schema_hash"] == schema.schema_hash()


def test_post_llm_spec_persists_to_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a successful LLM analyze, the spec is written to the disk cache."""
    from dbsprout.spec.cache import SpecCache  # noqa: PLC0415

    app = _make_app(tmp_path)
    schema = _small_schema()
    app.state.workspace.set_schema(schema)

    cache = SpecCache(cache_dir=tmp_path / "spec_cache")
    app.state.workspace.set_spec_cache(cache)

    llm_spec = _sentinel_spec(schema.schema_hash(), marker="from_llm_persisted")

    class _FakeAnalyzer:
        def __init__(self, provider: object) -> None:
            return

        def analyze(self, _schema: DatabaseSchema) -> DataSpec:
            return llm_spec

        def get_last_usage(self) -> None:
            return None

    class _FakeProvider:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return

    from dbsprout.spec import analyzer as analyzer_mod  # noqa: PLC0415
    from dbsprout.spec.providers import embedded as embedded_mod  # noqa: PLC0415

    monkeypatch.setattr(analyzer_mod, "SpecAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr(embedded_mod, "EmbeddedProvider", _FakeProvider)

    TestClient(app).post("/wizard/step/3/llm-spec")

    cached = cache.get(schema.schema_hash())
    assert cached is not None
    assert cached.model_used == "from_llm_persisted"


# ── errors.py — LLM_UNAVAILABLE enum + factory ───────────────────────────


def test_web_error_code_llm_unavailable_is_member() -> None:
    """The closed taxonomy now carries LLM_UNAVAILABLE."""
    from dbsprout.web.errors import WebErrorCode  # noqa: PLC0415

    assert WebErrorCode.LLM_UNAVAILABLE.value == "LLM_UNAVAILABLE"


def test_web_error_llm_unavailable_factory_shape() -> None:
    """Factory returns a 503 envelope with the reason in the message."""
    from dbsprout.web.errors import (  # noqa: PLC0415
        WebErrorCode,
        web_error_llm_unavailable,
    )

    err = web_error_llm_unavailable("llama-cpp-python not installed")

    assert err.code is WebErrorCode.LLM_UNAVAILABLE
    assert err.status_code == 503
    assert "llama-cpp-python not installed" in err.message
    assert err.hint is not None
