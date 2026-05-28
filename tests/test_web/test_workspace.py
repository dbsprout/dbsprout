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
from dbsprout.web.workspace import Workspace, _mask_userinfo, _redact_url


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


def test_redacted_target_masks_password_for_arbitrary_driver() -> None:
    """SQLAlchemy's ``make_url`` is lenient and still masks unusual schemes."""
    ws = Workspace()
    ws.set_target_url("mongodb+srv://bob:hunter2@cluster.example.com/x")
    redacted = ws.redacted_target
    assert redacted is not None
    assert "hunter2" not in redacted
    assert "cluster.example.com" in redacted


def test_redacted_target_no_password_unchanged() -> None:
    ws = Workspace()
    ws.set_target_url("sqlite:///local.db")
    assert ws.redacted_target == "sqlite:///local.db"


def test_clear_target_url() -> None:
    ws = Workspace()
    ws.set_target_url("postgresql://u:p@h/db")
    ws.clear_target_url()
    assert ws.redacted_target is None


# ── redaction helpers (direct, incl. the stdlib fallback) ─────────────


def test_redact_url_falls_back_when_sqlalchemy_cannot_parse() -> None:
    """A string SQLAlchemy rejects routes through the stdlib mask (never raises)."""
    # ``foo bar://`` has a space in the scheme → SQLAlchemy ``ArgumentError``;
    # urlsplit also can't extract userinfo from it, so it returns unchanged —
    # the point is the ``except`` branch is exercised and nothing leaks/raises.
    assert _redact_url("foo bar://u:pw@h/x") == "foo bar://u:pw@h/x"


def test_mask_userinfo_masks_password_with_port() -> None:
    assert _mask_userinfo("redis://user:secret@host:6379/0") == "redis://user:***@host:6379/0"


def test_mask_userinfo_masks_password_without_username() -> None:
    assert _mask_userinfo("redis://:secret@host/0") == "redis://***@host/0"


def test_mask_userinfo_passes_through_when_no_password() -> None:
    assert _mask_userinfo("redis://host/0") == "redis://host/0"


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


# ── repr never leaks credentials ──────────────────────────────────────


def test_repr_redacts_target_url() -> None:
    ws = Workspace()
    ws.set_target_url("postgresql://admin:SUPERSECRET@db:5432/prod")
    text = repr(ws)
    assert "SUPERSECRET" not in text
    assert "***" in text
    assert "db:5432/prod" in text


def test_repr_does_not_dump_schema_contents() -> None:
    ws = Workspace()
    ws.set_schema(_schema())
    assert repr(ws) == (
        "Workspace(schema=set, spec=None, last_result=None, source=None, target=None)"
    )


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


def test_peek_target_url_returns_raw_then_redacted_property_masks() -> None:
    """``peek_target_url`` exposes the raw URL (for internal credential scrubbing)
    while the public ``redacted_target`` property still masks the password."""
    ws = Workspace()
    assert ws.peek_target_url() is None
    raw = "postgresql://alice:s3cretpw@h:5432/db"
    ws.set_target_url(raw)
    assert ws.peek_target_url() == raw  # raw, for internal scrubbing only
    assert "s3cretpw" not in (ws.redacted_target or "")  # public accessor still masks


# ── reference-data seam (S-134) ───────────────────────────────────────


def test_reference_data_default_is_none() -> None:
    """A fresh workspace has no reference data; the fidelity/detection blocks
    in ``POST /api/validate`` rely on this default to degrade gracefully."""
    ws = Workspace()
    assert ws.get_reference_data() is None


def test_reference_data_round_trip() -> None:
    """``set_reference_data`` / ``get_reference_data`` round-trip the mapping
    that ``validate_fidelity`` / ``validate_detection`` consume."""
    ws = Workspace()
    payload = {"users": [{"id": 1, "email": "a@x"}, {"id": 2, "email": "b@x"}]}
    ws.set_reference_data(payload)
    assert ws.get_reference_data() == payload


def test_reset_clears_reference_data() -> None:
    """``reset()`` must clear reference data along with the rest of session state."""
    ws = Workspace()
    ws.set_reference_data({"users": [{"id": 1}]})
    ws.reset()
    assert ws.get_reference_data() is None
