"""FastAPI web dashboard application factory (S-090).

The dashboard is a *read-only* surface over the SQLite state layer
(``.dbsprout/state.db``, S-079). It never imports CLI/generation code and the
CLI never imports this module at startup — ``dbsprout serve`` lazy-imports it
(see :mod:`dbsprout.cli.serve`). FastAPI/uvicorn ship in the optional ``[web]``
extra.

:func:`create_app` is a factory (not a module-global singleton) so tests can
build isolated apps pointed at a temporary state DB. A module-level
``app = create_app()`` is exported too, so ``uvicorn dbsprout.web.app:app``
works for production serving.

The factory wires three things siblings rely on:

* ``app.state.templates`` — the shared :class:`~fastapi.templating.Jinja2Templates`
  environment (templates live in ``dbsprout/web/templates``).
* ``app.state.get_state_db`` — a zero-arg factory returning a fresh
  :class:`~dbsprout.state.db.StateDB` per request (cheap; opens a WAL connection).
* the shared :data:`~dbsprout.web.routes.router`, included via
  ``app.include_router`` — siblings append their handlers there.

Static assets (``dbsprout/web/static``) are mounted at ``/static``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dbsprout.migrate.snapshot import SnapshotStore
from dbsprout.state.db import StateDB
from dbsprout.web.jobs import JobManager, JobRecord
from dbsprout.web.progress import ProgressHub, progress_ws_router
from dbsprout.web.routers.connect import connect_router
from dbsprout.web.routers.generate import generate_router
from dbsprout.web.routers.jobs import jobs_router
from dbsprout.web.routers.preview import preview_router
from dbsprout.web.routers.schema import schema_router
from dbsprout.web.routers.schema_load import schema_load_router
from dbsprout.web.routers.studio import studio_router
from dbsprout.web.routes import router
from dbsprout.web.views.erd import erd_router
from dbsprout.web.views.insights import insights_router
from dbsprout.web.views.progress import progress_router
from dbsprout.web.workspace import Workspace

if TYPE_CHECKING:
    from dbsprout.state.models import RunRecord

_PACKAGE_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _PACKAGE_DIR / "templates"
_STATIC_DIR = _PACKAGE_DIR / "static"

#: Environment variable overriding the state-DB location (used by tests and by
#: anyone running the dashboard from outside the project root).
STATE_DB_ENV = "DBSPROUT_STATE_DB"

#: Default state-DB path, relative to the working directory.
DEFAULT_STATE_DB = Path(".dbsprout/state.db")

#: Environment variable overriding the schema-snapshot directory (S-091 ERD view).
SNAPSHOT_DIR_ENV = "DBSPROUT_SNAPSHOT_DIR"

#: Default snapshot directory, relative to the working directory.
DEFAULT_SNAPSHOT_DIR = Path(".dbsprout/snapshots")


def _resolve_state_db_path(state_db_path: Path | str | None) -> Path:
    """Pick the state-DB path: explicit arg > ``DBSPROUT_STATE_DB`` env > default."""
    if state_db_path is not None:
        return Path(state_db_path)
    env_value = os.environ.get(STATE_DB_ENV)
    if env_value:
        return Path(env_value)
    return DEFAULT_STATE_DB


def _resolve_snapshot_dir(snapshot_dir: Path | str | None) -> Path:
    """Pick the snapshot dir: explicit arg > ``DBSPROUT_SNAPSHOT_DIR`` env > default."""
    if snapshot_dir is not None:
        return Path(snapshot_dir)
    env_value = os.environ.get(SNAPSHOT_DIR_ENV)
    if env_value:
        return Path(env_value)
    return DEFAULT_SNAPSHOT_DIR


def create_app(
    state_db_path: Path | str | None = None,
    snapshot_dir: Path | str | None = None,
) -> FastAPI:
    """Build a configured FastAPI dashboard app.

    *state_db_path* overrides where run telemetry is read from; when ``None``
    the ``DBSPROUT_STATE_DB`` env var (then :data:`DEFAULT_STATE_DB`) is used.

    *snapshot_dir* overrides where schema snapshots are read from (S-091 ERD
    view); when ``None`` the ``DBSPROUT_SNAPSHOT_DIR`` env var (then
    :data:`DEFAULT_SNAPSHOT_DIR`) is used.
    """
    resolved = _resolve_state_db_path(state_db_path)
    resolved_snapshots = _resolve_snapshot_dir(snapshot_dir)

    app = FastAPI(
        title="DBSprout Dashboard",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    app.state.get_state_db = lambda: StateDB(resolved)
    app.state.get_snapshot_store = lambda: SnapshotStore(base_dir=resolved_snapshots)
    # ─── S-111 in-memory session ───
    # Single-user localhost → exactly one Workspace instance per app, shared
    # across stateless requests. No write routes yet (later stories).
    app.state.workspace = Workspace()
    # ─── end S-111 ───
    # ─── S-108 job manager ───
    # Single active background job (generate) for the single-user localhost
    # dashboard. The job manager publishes progress to the S-109 hub (below).
    # ─── end S-108 ───
    # ── S-109 progress ws ──
    # Live per-table progress over a WebSocket. One ProgressHub bridges the
    # JobManager's worker-thread ProgressEvents to subscribed WS clients; the
    # manager and hub share one instance so events published on the worker
    # thread reach GET /ws/jobs/{job_id}.
    progress_hub = ProgressHub()
    app.state.progress_hub = progress_hub

    # ── S-110 state-write hook ──
    # On SUCCEEDED completion the JobManager calls this closure with the
    # final JobRecord; we lazy-import the writer (preserving the
    # ``dbsprout serve`` lazy-import contract for the manager module
    # itself) and persist a RunRecord against the resolved state DB.
    # FAILED / CANCELLED jobs are NOT written (the manager guards). A
    # raise from the writer is swallowed (best-effort telemetry).
    def _persist_completed_job(record: JobRecord) -> None:
        from dbsprout.generate.orchestrator import GenerateResult  # noqa: PLC0415
        from dbsprout.state.writer import record_job_run  # noqa: PLC0415

        if not isinstance(record.result, GenerateResult):
            return  # opaque/non-generate result — nothing meaningful to persist
        record_job_run(
            record.result,
            engine=record.engine or "heuristic",
            seed=record.seed if record.seed is not None else 0,
            started_at=record.started_at,
            completed_at=record.finished_at,
            db_path=resolved,
        )

    def _read_persisted_runs() -> list[RunRecord]:
        return StateDB(resolved).get_runs()

    app.state.job_manager = JobManager(
        progress_hub=progress_hub,
        state_writer=_persist_completed_job,
        state_reader=_read_persisted_runs,
    )
    app.include_router(progress_ws_router)
    # ── end S-109 ──
    # ── end S-110 ──
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    app.include_router(router)
    # ─── S-091 ERD region ───
    app.include_router(erd_router)
    # ─── end S-091 ───
    # ─── S-092 progress region ───
    app.include_router(progress_router)
    # ─── end S-092 ───
    # ─── S-093 views region ───
    # Real /quality view (S-090's placeholder was removed from routes.py) plus
    # /preview, /preview/{table}, /costs and /history — all read-only over the
    # state DB.
    app.include_router(insights_router)
    # ─── end S-093 ───
    # ── S-112 connect router ──
    # First read-WRITE JSON API: POST /api/connect introspects a live DB via the
    # core-service facade and stores the schema + redacted target on the
    # workspace. Lives under dbsprout/web/routers/ (write APIs), distinct from
    # the read-only views/ above.
    app.include_router(connect_router)
    # ── end S-112 ──
    # ── S-113 schema-load router ──
    # POST /api/schema/load — multipart upload parsed via the existing parsers,
    # result stored on app.state.workspace (S-111).
    app.include_router(schema_load_router)
    # ── end S-113 ──
    # ── S-124 generate route ──
    # POST /api/generate — submits the generation pipeline as a background job
    # via app.state.job_manager (S-108), running core.service.generate (S-106)
    # over the schema in app.state.workspace (S-111); returns {job_id} at once.
    app.include_router(generate_router)
    # ── end S-124 ──
    # ── S-126 cancel route ──
    # POST /api/jobs/{job_id}/cancel — cooperative cancel for the active
    # generate job; arms the S-107 cancel_token on the JobManager (S-108).
    # Lives in its own module so the S-125 parallel branch (Studio console
    # live tail) does not collide on the generate router.
    app.include_router(jobs_router)
    # ── end S-126 ──
    # ── S-147 preview route ──
    # GET /api/preview/{table}?limit=N — bounded JSON sample of generated rows
    # from app.state.workspace.last_result (populated by POST /api/generate,
    # S-124). Read-only; powers the Studio grid's preview/regen feedback loop.
    app.include_router(preview_router)
    # ── end S-147 ──
    # ── S-115 schema view ──
    # Read-only workspace review: GET /api/schema (tree JSON) + GET
    # /api/schema/erd (HTMX ERD fragment, reusing build_erd_mermaid). Distinct
    # from the snapshot-backed GET /schema view; full Studio layout is S-117.
    app.include_router(schema_router)
    # ── end S-115 ──
    # ── S-117 studio shell ──
    # GET /studio — single-page workspace shell with four named-slot panels
    # (tree · grid · context · console). Later Phase-C stories (S-118 spec
    # grid, S-125 console progress, S-127 seed control) plug into the stable
    # element ids (#studio-tree / #studio-grid / #studio-context /
    # #studio-console) without editing this shell.
    app.include_router(studio_router)
    # ── end S-117 ──
    return app


app = create_app()
