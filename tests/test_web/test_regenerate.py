"""POST /api/regenerate sync/job route tests (S-131).

The endpoint re-runs generation for one column or one whole table, choosing
between an inline (sync) call and a background job based on a row-count
threshold. The web stack lives in the optional ``[web]`` extra, so the module
guards with ``pytest.importorskip("fastapi")`` before importing FastAPI
symbols.

Tests seed ``app.state.workspace`` directly (schema + last_result + seed)
rather than going through ``POST /api/connect`` + ``POST /api/generate`` — the
upstream routes have their own dedicated tests, and this keeps each unit small.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.generate.orchestrator import GenerateResult
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


# ── helpers ─────────────────────────────────────────────────────────────


def _make_app(tmp_path: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=tmp_path / "state.db")


def _int_col(name: str, *, nullable: bool = True, pk: bool = False) -> ColumnSchema:
    return ColumnSchema(
        name=name,
        data_type=ColumnType.INTEGER,
        nullable=nullable,
        primary_key=pk,
    )


def _str_col(name: str, *, nullable: bool = True) -> ColumnSchema:
    return ColumnSchema(
        name=name,
        data_type=ColumnType.VARCHAR,
        nullable=nullable,
        max_length=50,
    )


def _schema_two_tables() -> DatabaseSchema:
    parent = TableSchema(
        name="departments",
        columns=[
            _int_col("id", nullable=False, pk=True),
            _str_col("name", nullable=False),
        ],
        primary_key=["id"],
    )
    child = TableSchema(
        name="employees",
        columns=[
            _int_col("id", nullable=False, pk=True),
            _str_col("name", nullable=False),
            _str_col("email", nullable=False),
            _int_col("dept_id", nullable=True),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(
                columns=["dept_id"],
                ref_table="departments",
                ref_columns=["id"],
            ),
        ],
    )
    return DatabaseSchema(tables=[parent, child], dialect="sqlite")


def _initial_state() -> dict[str, list[dict[str, Any]]]:
    parents = [{"id": i, "name": f"D{i}"} for i in (10, 20, 30)]
    children = [
        {"id": 1, "name": "Alice", "email": "alice@a.com", "dept_id": 10},
        {"id": 2, "name": "Bob", "email": "bob@a.com", "dept_id": 20},
        {"id": 3, "name": "Carol", "email": "carol@a.com", "dept_id": 30},
        {"id": 4, "name": "Dan", "email": "dan@a.com", "dept_id": 10},
        {"id": 5, "name": "Eve", "email": "eve@a.com", "dept_id": 20},
    ]
    return {"departments": parents, "employees": children}


def _seed_workspace(
    app: FastAPI,
    *,
    schema: DatabaseSchema | None = None,
    state: dict[str, list[dict[str, Any]]] | None = None,
    seed: int = 42,
) -> None:
    workspace = app.state.workspace
    if schema is not None:
        workspace.set_schema(schema)
    if state is not None:
        result = GenerateResult(
            tables_data=state,
            insertion_order=list(state.keys()),
            total_rows=sum(len(rows) for rows in state.values()),
            total_tables=len(state),
            duration_seconds=0.0,
        )
        workspace.set_last_result(result)
        workspace.set_last_seed(seed)


# ── router-registration smoke ───────────────────────────────────────────


def test_regenerate_router_is_importable() -> None:
    from fastapi import APIRouter  # noqa: PLC0415

    from dbsprout.web.routers.regenerate import regenerate_router  # noqa: PLC0415

    assert isinstance(regenerate_router, APIRouter)


def test_regenerate_route_is_mounted(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/api/regenerate" in paths


# ── request validation ─────────────────────────────────────────────────


def test_regenerate_missing_table_is_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, schema=_schema_two_tables(), state=_initial_state())
    resp = TestClient(app).post("/api/regenerate", json={})
    assert resp.status_code == 422, resp.text


def test_regenerate_extra_field_is_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, schema=_schema_two_tables(), state=_initial_state())
    resp = TestClient(app).post(
        "/api/regenerate",
        json={"table": "employees", "unexpected": 1},
    )
    assert resp.status_code == 422, resp.text


def test_regenerate_negative_reroll_is_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, schema=_schema_two_tables(), state=_initial_state())
    resp = TestClient(app).post(
        "/api/regenerate",
        json={"table": "employees", "reroll": -1},
    )
    assert resp.status_code == 422, resp.text


# ── no-schema / no-run guards ──────────────────────────────────────────


def test_regenerate_no_schema_is_409(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    resp = TestClient(app).post("/api/regenerate", json={"table": "employees"})
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "NO_SCHEMA"
    assert "Traceback" not in resp.text


def test_regenerate_no_last_result_is_409(tmp_path: Path) -> None:
    """Schema loaded but no generation has run yet → 409 NO_RUN."""
    app = _make_app(tmp_path)
    _seed_workspace(app, schema=_schema_two_tables())
    resp = TestClient(app).post("/api/regenerate", json={"table": "employees"})
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "NO_RUN"


# ── unknown table / column → 404 NOT_FOUND ─────────────────────────────


def test_regenerate_unknown_table_is_404(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, schema=_schema_two_tables(), state=_initial_state())
    resp = TestClient(app).post("/api/regenerate", json={"table": "missing"})
    assert resp.status_code == 404, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "NOT_FOUND"
    assert "missing" in detail["message"]


def test_regenerate_unknown_column_is_404(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, schema=_schema_two_tables(), state=_initial_state())
    resp = TestClient(app).post(
        "/api/regenerate",
        json={"table": "employees", "column": "missing"},
    )
    assert resp.status_code == 404, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "NOT_FOUND"


# ── PK / FK-referenced → 409 CONSTRAINT_VIOLATION ──────────────────────


def test_regenerate_column_pk_is_409_constraint_violation(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, schema=_schema_two_tables(), state=_initial_state())
    resp = TestClient(app).post(
        "/api/regenerate",
        json={"table": "employees", "column": "id"},
    )
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "CONSTRAINT_VIOLATION"


def test_regenerate_column_fk_referenced_is_409_constraint_violation(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, schema=_schema_two_tables(), state=_initial_state())
    resp = TestClient(app).post(
        "/api/regenerate",
        json={"table": "departments", "column": "id"},
    )
    # departments.id is PK and FK-referenced; the PK check fires first but
    # both are CONSTRAINT_VIOLATION, so the assert remains stable.
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "CONSTRAINT_VIOLATION"


# ── sync happy path: whole-table regen ──────────────────────────────────


def test_regenerate_table_sync_returns_rows_affected(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    state = _initial_state()
    _seed_workspace(app, schema=_schema_two_tables(), state=state)
    resp = TestClient(app).post("/api/regenerate", json={"table": "employees"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "sync"
    assert body["table"] == "employees"
    assert body["column"] is None
    assert body["rows_affected"] == 5
    # PKs preserved
    new_ids = [r["id"] for r in body["rows"]]
    assert new_ids == [1, 2, 3, 4, 5]


def test_regenerate_table_sync_updates_workspace_last_result(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    state = _initial_state()
    _seed_workspace(app, schema=_schema_two_tables(), state=state)
    client = TestClient(app)
    r1 = client.post("/api/regenerate", json={"table": "employees"})
    assert r1.status_code == 200
    workspace = app.state.workspace
    updated = workspace.get_last_result()
    assert updated is not None
    # PKs preserved + same length
    assert [r["id"] for r in updated.tables_data["employees"]] == [1, 2, 3, 4, 5]
    # Untouched table is byte-identical
    assert updated.tables_data["departments"] == state["departments"]


# ── sync happy path: single-column regen ───────────────────────────────


def test_regenerate_column_sync_only_changes_target_column(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    state = _initial_state()
    _seed_workspace(app, schema=_schema_two_tables(), state=state)
    resp = TestClient(app).post(
        "/api/regenerate",
        json={"table": "employees", "column": "name"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "sync"
    assert body["table"] == "employees"
    assert body["column"] == "name"
    assert body["rows_affected"] == 5
    # PK + other columns identical for every row
    for orig, new in zip(state["employees"], body["rows"], strict=True):
        assert new["id"] == orig["id"]
        assert new["email"] == orig["email"]
        assert new["dept_id"] == orig["dept_id"]


def test_regenerate_column_sync_reroll_changes_output(tmp_path: Path) -> None:
    """Same seed + different reroll nonce yields a different draw."""
    app = _make_app(tmp_path)
    _seed_workspace(app, schema=_schema_two_tables(), state=_initial_state())
    client = TestClient(app)
    r1 = client.post(
        "/api/regenerate",
        json={"table": "employees", "column": "name", "reroll": 0},
    )
    # reset state so we are not regenerating from already-regenerated state
    _seed_workspace(app, schema=_schema_two_tables(), state=_initial_state())
    r2 = client.post(
        "/api/regenerate",
        json={"table": "employees", "column": "name", "reroll": 1},
    )
    assert r1.status_code == 200
    assert r2.status_code == 200
    names1 = [r["name"] for r in r1.json()["rows"]]
    names2 = [r["name"] for r in r2.json()["rows"]]
    assert names1 != names2


def test_regenerate_column_sync_same_reroll_is_deterministic(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    _seed_workspace(app, schema=_schema_two_tables(), state=_initial_state())
    client = TestClient(app)
    r1 = client.post(
        "/api/regenerate",
        json={"table": "employees", "column": "name", "reroll": 7},
    )
    _seed_workspace(app, schema=_schema_two_tables(), state=_initial_state())
    r2 = client.post(
        "/api/regenerate",
        json={"table": "employees", "column": "name", "reroll": 7},
    )
    assert r1.json()["rows"] == r2.json()["rows"]


# ── job path: rows above threshold ────────────────────────────────────


def test_regenerate_above_threshold_enqueues_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When affected rows exceed SYNC_REGEN_THRESHOLD the route enqueues a job."""
    # Lower the threshold to make the test fast.
    monkeypatch.setattr(
        "dbsprout.web.routers.regenerate.SYNC_REGEN_THRESHOLD",
        3,
    )
    app = _make_app(tmp_path)
    _seed_workspace(app, schema=_schema_two_tables(), state=_initial_state())
    resp = TestClient(app).post("/api/regenerate", json={"table": "employees"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "job"
    assert isinstance(body["job_id"], str)
    assert body["status"] == "running"


def test_regenerate_job_path_skips_when_no_last_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The job-path branch still falls through the no-run guard."""
    monkeypatch.setattr(
        "dbsprout.web.routers.regenerate.SYNC_REGEN_THRESHOLD",
        0,
    )
    app = _make_app(tmp_path)
    _seed_workspace(app, schema=_schema_two_tables())
    resp = TestClient(app).post("/api/regenerate", json={"table": "employees"})
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["code"] == "NO_RUN"


def test_regenerate_job_closure_runs_to_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the background job closure directly and check the JobRecord.

    The TestClient runs the ASGI app in its own thread, so ``asyncio.Task``\\s
    spawned by ``submit`` cannot be reliably joined from the calling test
    thread (the loop dies with the request). Instead, this test calls the
    route's job-closure builder + the manager directly under ``anyio.run`` —
    the underlying logic (progress events, workspace update, RegenerateError
    mapping) is identical to what the route triggers.
    """
    import anyio  # noqa: PLC0415

    monkeypatch.setattr("dbsprout.web.routers.regenerate.SYNC_REGEN_THRESHOLD", 3)
    app = _make_app(tmp_path)
    _seed_workspace(app, schema=_schema_two_tables(), state=_initial_state())

    from dbsprout.web.routers.regenerate import (  # noqa: PLC0415
        RegenerateRequest,
        _build_job_fn,
    )

    body = RegenerateRequest(table="employees")

    async def _drive() -> str:
        # Build the closure exactly the way the route does (the ``request``
        # arg is only used inside the ``_call_regenerate`` error path — fine
        # to pass a sentinel since we drive the happy path here).
        fn = _build_job_fn(body, app.state.workspace, request=None)  # type: ignore[arg-type]
        new_job_id = await app.state.job_manager.submit("regenerate", fn)
        await app.state.job_manager.wait(new_job_id)
        return new_job_id

    job_id = anyio.run(_drive)

    record = app.state.job_manager.get(job_id)
    assert record.status.value == "succeeded", record.error
    assert isinstance(record.result, dict)
    assert record.result["table"] == "employees"
    assert record.result["rows_affected"] == 5
    phases = [e.phase for e in record.events]
    assert "table_start" in phases
    assert "table_done" in phases
    updated = app.state.workspace.get_last_result()
    assert updated is not None
    assert len(updated.tables_data["employees"]) == 5


def test_regenerate_job_closure_respects_cancel_token(tmp_path: Path) -> None:
    """A pre-cancelled token raises GenerationCancelled at the closure's top.

    Exercises the cancel branch of the job closure (the first
    ``_is_cancelled`` check).
    """
    from dbsprout.generate.progress import GenerationCancelled  # noqa: PLC0415
    from dbsprout.web.routers.regenerate import (  # noqa: PLC0415
        RegenerateRequest,
        _build_job_fn,
    )

    app = _make_app(tmp_path)
    _seed_workspace(app, schema=_schema_two_tables(), state=_initial_state())

    class _PreCancelled:
        def is_cancelled(self) -> bool:
            return True

    body = RegenerateRequest(table="employees")
    fn = _build_job_fn(body, app.state.workspace, request=None)  # type: ignore[arg-type]
    with pytest.raises(GenerationCancelled):
        fn(lambda _e: None, _PreCancelled())  # type: ignore[arg-type]


def test_regenerate_no_rows_in_table_is_409(tmp_path: Path) -> None:
    """A table with 0 rows in the state surfaces as 409 NO_RUN.

    The route's row-count guard passes (the table key exists in
    ``last_result.tables_data``) but the underlying ``regenerate_table``
    raises ``RegenerateError(reason="no_rows")`` — the mapper translates that
    to the friendly NO_RUN envelope.
    """
    app = _make_app(tmp_path)
    state = {"departments": [], "employees": []}
    _seed_workspace(app, schema=_schema_two_tables(), state=state)
    # The above sets last_result with empty rows; tables_data is non-empty
    # (the keys exist) so the route falls through to the regen entry point.
    resp = TestClient(app).post("/api/regenerate", json={"table": "employees"})
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["code"] == "NO_RUN"


def test_regenerate_uses_fallback_seed_when_workspace_has_none(tmp_path: Path) -> None:
    """When ``workspace.last_seed is None`` the route falls back to ``_FALLBACK_SEED``.

    The test seeds ``last_result`` without recording a seed — same call twice
    should be byte-identical (fallback is deterministic).
    """
    app = _make_app(tmp_path)
    schema = _schema_two_tables()
    state = _initial_state()
    workspace = app.state.workspace
    workspace.set_schema(schema)
    from dbsprout.generate.orchestrator import GenerateResult  # noqa: PLC0415

    workspace.set_last_result(
        GenerateResult(
            tables_data=state,
            insertion_order=list(state.keys()),
            total_rows=sum(len(rows) for rows in state.values()),
            total_tables=len(state),
            duration_seconds=0.0,
        ),
    )
    # last_seed stays None — exercises the fallback branch.
    assert workspace.get_last_seed() is None
    client = TestClient(app)
    r1 = client.post(
        "/api/regenerate",
        json={"table": "employees", "column": "name", "reroll": 0},
    )
    assert r1.status_code == 200
    # Re-seed the state to wipe the regen, ask again with the same nonce.
    workspace.set_last_result(
        GenerateResult(
            tables_data=_initial_state(),
            insertion_order=["departments", "employees"],
            total_rows=8,
            total_tables=2,
            duration_seconds=0.0,
        ),
    )
    r2 = client.post(
        "/api/regenerate",
        json={"table": "employees", "column": "name", "reroll": 0},
    )
    assert r2.status_code == 200
    assert r1.json()["rows"] == r2.json()["rows"]


def test_regenerate_job_path_conflict_when_already_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second submit while a job is active is mapped to a friendly 409.

    Stubs ``app.state.job_manager.submit`` to raise ``JobError`` so the test
    does not need a real long-running job in flight.
    """
    monkeypatch.setattr("dbsprout.web.routers.regenerate.SYNC_REGEN_THRESHOLD", 0)
    app = _make_app(tmp_path)
    _seed_workspace(app, schema=_schema_two_tables(), state=_initial_state())

    from dbsprout.web.jobs import JobError  # noqa: PLC0415

    async def _raise_busy(*_args: Any, **_kwargs: Any) -> str:
        raise JobError("a job is already running")

    monkeypatch.setattr(app.state.job_manager, "submit", _raise_busy)
    resp = TestClient(app).post("/api/regenerate", json={"table": "employees"})
    assert resp.status_code == 409, resp.text
    assert "already running" in str(resp.json()["detail"])


# ── lazy-import contract ──────────────────────────────────────────────


def test_regenerate_router_no_eager_regenerate_import() -> None:
    """Importing the router must not pull dbsprout.generate.regenerate eagerly."""
    probe = (
        "import sys\n"
        "import dbsprout.web.routers.regenerate  # noqa: F401\n"
        "bad = [m for m in ('dbsprout.generate.regenerate',) if m in sys.modules]\n"
        "print(bad)\n"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, trusted interpreter
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == "[]", result.stdout.strip()
