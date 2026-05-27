"""Workspace in-memory session tests (S-111).

The web stack lives in the optional ``[web]`` extra; the ``Workspace`` object
itself needs no FastAPI, but it ships under the web package, so the module-level
``importorskip`` mirrors the sibling web tests. The ``app.state`` wiring cases
exercise ``create_app`` + ``TestClient``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.generate.orchestrator import GenerateResult
from dbsprout.schema.models import ColumnSchema, ColumnType, DatabaseSchema, TableSchema
from dbsprout.spec.models import DataSpec, GeneratorConfig, TableSpec
from dbsprout.web.app import create_app
from dbsprout.web.workspace import Workspace


def _schema() -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="users",
                columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True)],
                primary_key=["id"],
            )
        ]
    )


def _spec() -> DataSpec:
    return DataSpec(
        tables=[
            TableSpec(
                table_name="users",
                columns={"id": GeneratorConfig(provider="numeric", method="sequence")},
            )
        ],
        global_seed=42,
    )


# ── defaults ──────────────────────────────────────────────────────────


def test_fresh_workspace_is_empty() -> None:
    ws = Workspace()
    assert ws.get_schema() is None
    assert ws.get_spec() is None
    assert ws.get_last_result() is None
    assert ws.get_source() is None
    assert ws.redacted_target is None


# ── set / get round-trips ─────────────────────────────────────────────


def test_set_get_schema_spec_result_source() -> None:
    ws = Workspace()
    schema, spec = _schema(), _spec()
    result = GenerateResult(total_rows=5, total_tables=1)

    ws.set_schema(schema)
    ws.set_spec(spec)
    ws.set_last_result(result)
    ws.set_source("file: schema.sql")

    assert ws.get_schema() is schema
    assert ws.get_spec() is spec
    assert ws.get_last_result() is result
    assert ws.get_source() == "file: schema.sql"


# ── immutable spec edits ──────────────────────────────────────────────


def test_update_spec_returns_new_object_and_does_not_mutate_original() -> None:
    ws = Workspace()
    original = _spec()
    ws.set_spec(original)

    updated = ws.update_spec(global_seed=99)

    assert updated is not original  # new object (model_copy)
    assert updated.global_seed == 99
    assert original.global_seed == 42  # frozen original untouched
    assert ws.get_spec() is updated  # stored copy is the new one


def test_update_spec_without_loaded_spec_raises() -> None:
    ws = Workspace()
    with pytest.raises(ValueError, match="no spec loaded"):
        ws.update_spec(global_seed=1)


# ── credential redaction ──────────────────────────────────────────────


def test_redacted_target_masks_password() -> None:
    ws = Workspace()
    ws.set_target_url("postgresql+psycopg://alice:s3cret@db.host:5432/app")
    redacted = ws.redacted_target
    assert redacted is not None
    assert "s3cret" not in redacted
    assert "db.host" in redacted
    assert "app" in redacted


def test_redacted_target_fallback_masks_non_sqlalchemy_url() -> None:
    ws = Workspace()
    ws.set_target_url("weird://bob:hunter2@example.com/x")
    redacted = ws.redacted_target
    assert redacted is not None
    assert "hunter2" not in redacted
    assert "example.com" in redacted


def test_redacted_target_no_password_unchanged() -> None:
    ws = Workspace()
    ws.set_target_url("sqlite:///local.db")
    assert ws.redacted_target == "sqlite:///local.db"


def test_clear_target_url() -> None:
    ws = Workspace()
    ws.set_target_url("postgresql://u:p@h/db")
    ws.clear_target_url()
    assert ws.redacted_target is None


# ── reset ─────────────────────────────────────────────────────────────


def test_reset_clears_all_state() -> None:
    ws = Workspace()
    ws.set_schema(_schema())
    ws.set_spec(_spec())
    ws.set_last_result(GenerateResult(total_rows=1))
    ws.set_source("db: x")
    ws.set_target_url("postgresql://u:p@h/db")

    ws.reset()

    assert ws.get_schema() is None
    assert ws.get_spec() is None
    assert ws.get_last_result() is None
    assert ws.get_source() is None
    assert ws.redacted_target is None


# ── app.state wiring ──────────────────────────────────────────────────


def test_create_app_wires_single_workspace_on_state() -> None:
    app = create_app()
    assert isinstance(app.state.workspace, Workspace)


def test_workspace_is_shared_across_requests() -> None:
    app = create_app()
    app.state.workspace.set_source("file: shared.sql")
    with TestClient(app) as client:
        client.get("/")  # any request; state persists on app, not request
    assert app.state.workspace.get_source() == "file: shared.sql"


def test_separate_apps_get_independent_workspaces() -> None:
    app_a = create_app()
    app_b = create_app()
    assert app_a.state.workspace is not app_b.state.workspace


def test_workspace_wiring_adds_no_routes() -> None:
    """The session object is state, not endpoints — no route mentions 'workspace'."""
    app = create_app()
    paths = [getattr(r, "path", "") for r in app.routes]
    assert not any("workspace" in p for p in paths)
