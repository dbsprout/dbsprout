"""FastAPI web application factory (S-090; P1c-5 cutover).

The web server now exposes exactly two things: the React Workbench SPA at
``/app`` and the JSON ``/api/*`` data API (plus the progress WebSocket). The
legacy server-rendered HTMX/Alpine dashboard — wizard, studio, insights views,
ERD page, progress page, and their Jinja2 templates / static assets — was
removed in the P1c-5 cutover (design §7). ``GET /`` now redirects to ``/app``.

It never imports CLI/generation code and the CLI never imports this module at
startup — ``dbsprout serve`` lazy-imports it (see :mod:`dbsprout.cli.serve`).
FastAPI/uvicorn ship in the optional ``[web]`` extra.

:func:`create_app` is a factory (not a module-global singleton) so tests can
build isolated apps pointed at a temporary state DB. A module-level
``app = create_app()`` is exported too, so ``uvicorn dbsprout.web.app:app``
works for production serving.

The factory wires:

* ``app.state.get_state_db`` — a zero-arg factory returning a fresh
  :class:`~dbsprout.state.db.StateDB` per request (cheap; opens a WAL connection).
* ``app.state.workspace`` / ``app.state.job_manager`` / ``app.state.progress_hub``
  — the in-memory session, background-job manager, and live-progress hub.
* the SPA mount (:func:`~dbsprout.web.spa.mount_spa`), every ``/api/*`` JSON
  router, and the progress WebSocket router.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from dbsprout.migrate.snapshot import SnapshotStore
from dbsprout.state.db import StateDB
from dbsprout.web.jobs import JobManager, JobRecord
from dbsprout.web.progress import ProgressHub, progress_ws_router
from dbsprout.web.routers.connect import connect_router
from dbsprout.web.routers.connections import connections_router
from dbsprout.web.routers.export import export_router
from dbsprout.web.routers.generate import generate_router
from dbsprout.web.routers.generators import generators_router
from dbsprout.web.routers.insert import insert_router
from dbsprout.web.routers.insights_api import insights_api_router
from dbsprout.web.routers.jobs import jobs_router
from dbsprout.web.routers.preview import preview_router
from dbsprout.web.routers.regenerate import regenerate_router
from dbsprout.web.routers.samples import samples_router
from dbsprout.web.routers.schema import schema_router
from dbsprout.web.routers.schema_load import schema_load_router
from dbsprout.web.routers.spec import spec_router
from dbsprout.web.routers.spec_assist import spec_assist_router
from dbsprout.web.routers.validate import validate_router
from dbsprout.web.routes import router
from dbsprout.web.spa import mount_spa
from dbsprout.web.workspace import Workspace

if TYPE_CHECKING:
    from dbsprout.state.models import RunRecord

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
        title="DBSprout Workbench",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
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
    # ── P1c-5 cutover: `/` redirects to the SPA; `/app` is the sole front door.
    @app.get("/", include_in_schema=False)
    async def _root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/app", status_code=308)

    # Serve the React Workbench SPA at /app (placeholder page if unbuilt).
    mount_spa(app)
    # The shared router now carries only the /health probe (the legacy index
    # dashboard was removed in the P1c-5 cutover).
    app.include_router(router)
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
    # ── P1a samples ──
    app.include_router(samples_router)
    # ── end P1a samples ──
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
    # ── S-136 insert route ──
    # POST /api/insert — submits a dialect-aware insertion run as a background
    # job via app.state.job_manager (S-108) over the data in
    # app.state.workspace.last_result (populated by POST /api/generate, S-124),
    # using the existing writers in dbsprout/output/ (PG COPY · MySQL LOAD DATA
    # · SaBatch fallback). Per-table progress streams on /ws/jobs/{job_id} (S-109).
    # Write-guard (S-137) seam lives in the same module; S-136 enforces token
    # presence (403 WRITE_GUARD_REQUIRED) so the write path is closed from day one.
    app.include_router(insert_router)
    # ── end S-136 ──
    # ── S-115 schema view ──
    # Read-only workspace review: GET /api/schema (tree JSON) + GET
    # /api/schema/erd (ERD JSON — mermaid string + table_details — for the SPA
    # to render client-side with Mermaid.js).
    app.include_router(schema_router)
    # ── end S-115 ──
    # ── S-118 spec router ──
    # GET /api/spec — DataSpec read endpoint over app.state.workspace (S-111),
    # building a heuristic spec lazily via spec.analyzer.heuristic_fallback
    # when no spec is cached. JSON-only. PUT /api/spec/tables/{t} and
    # PUT /api/spec/tables/{t}/columns/{c} edit the spec (JSON).
    app.include_router(spec_router)
    # ── end S-118 ──
    # ─── S-120 generators region ───
    # GET /api/generators — provider/method catalogue derived from the
    # spec.catalog helpers (heuristic PATTERNS + _TYPE_FALLBACKS). Read-only,
    # workspace-independent; the SPA method-picker fetches this once on
    # open to populate the dropdown.
    app.include_router(generators_router)
    # ─── end S-120 ───
    # ── S-131 regenerate router ──
    # POST /api/regenerate — re-roll one column or one whole table via the
    # surgical S-128/S-129 entry points in dbsprout/generate/regenerate.py.
    # Sync path when affected rows <= SYNC_REGEN_THRESHOLD (100_000); above
    # that the regen runs as a background job via app.state.job_manager
    # (S-108) and progress streams over the existing /ws/jobs/{job_id} (S-109).
    # Typed envelopes via dbsprout/web/errors.py — PK/FK-referenced regen is
    # rejected as 409 CONSTRAINT_VIOLATION, unknown table/column as 404
    # NOT_FOUND, no schema as 409 NO_SCHEMA.
    app.include_router(regenerate_router)
    # ── end S-131 ──
    # ── S-133 validate router ──
    # POST /api/validate — integrity + fidelity + detection report over
    # app.state.workspace.last_result. Reuses dbsprout/quality/* . 409 NO_RUN
    # when no generation has run yet. JSON-only.
    app.include_router(validate_router)
    # ── end S-133 ──
    # ── S-140 export route ──
    # POST /api/export — stream the workspace's last GenerateResult (S-111 / S-124)
    # back to the browser as a single file download. Resolves the writer through
    # dbsprout/plugins/dispatch::resolve_writer (same path the CLI uses) and
    # cleans the per-request temp dir from within the streaming generator. The
    # mount lives inside a delimited region block so S-142 (wizard) can mount
    # alongside without colliding on the same line.
    app.include_router(export_router)
    # ── end S-140 ──
    # ─── P1c-4 region ───
    # GET /api/runs · /api/quality · /api/costs — read-only JSON for the React
    # Workbench "Runs & Quality" panels. Wrap the pure builders (paginate_runs ·
    # build_quality_table · build_cost_summary) over the state DB; never leak
    # config_json/secrets; honest empty payloads (200) when the CLI never ran.
    app.include_router(insights_api_router)
    # ─── end P1c-4 region ───
    # ─── P2a-2 region ───
    # GET/POST/DELETE /api/connections — saved connections persisted to
    # .dbsprout/connections.toml via the pure dbsprout/core/connections.py
    # helper. Passwords are NEVER written: the stored value is empty or an
    # ${ENV_VAR} reference, resolved only at connect time.
    app.include_router(connections_router)
    # ─── end P2a-2 region ───
    # ─── P2b-3 region ───
    # POST /api/spec/assist — let an LLM propose a full DataSpec for the loaded
    # schema (offline EmbeddedProvider by default; opt-in CloudProvider). The
    # proposal is stored on app.state.workspace so GET /api/spec + the configure
    # grid reflect it, and cached by schema_hash. Provider modules (llama-cpp /
    # litellm) are lazy-imported inside the handler, so importing this router
    # pulls neither optional extra; a missing extra / model / key degrades to a
    # typed 503 LLM_UNAVAILABLE envelope (never a 500).
    app.include_router(spec_assist_router)
    # ─── end P2b-3 region ───
    return app


app = create_app()
