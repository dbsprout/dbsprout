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
from dbsprout.web.routers.export import export_router
from dbsprout.web.routers.generate import generate_router
from dbsprout.web.routers.generators import generators_router
from dbsprout.web.routers.insert import insert_router
from dbsprout.web.routers.jobs import jobs_router
from dbsprout.web.routers.preview import preview_router
from dbsprout.web.routers.regenerate import regenerate_router
from dbsprout.web.routers.samples import samples_router
from dbsprout.web.routers.schema import schema_router
from dbsprout.web.routers.schema_load import schema_load_router
from dbsprout.web.routers.spec import spec_router
from dbsprout.web.routers.studio import studio_router
from dbsprout.web.routers.validate import validate_router
from dbsprout.web.routers.wizard import wizard_router
from dbsprout.web.routes import router
from dbsprout.web.spa import mount_spa
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
    # ─── S-123 inline tooltips ───
    # Expose two helpers as Jinja globals so any template can render a
    # generator's human description or the GeneratorConfig field tooltips
    # without re-importing the spec module. Single source of truth: the
    # S-120 catalogue + the Pydantic ``description=`` fields on
    # ``GeneratorConfig``.
    from dbsprout.spec.catalog import _describe as _catalog_describe  # noqa: PLC0415
    from dbsprout.spec.models import field_descriptions  # noqa: PLC0415

    app.state.templates.env.globals["describe_method"] = _catalog_describe
    app.state.templates.env.globals["field_descriptions"] = field_descriptions
    # ─── end S-123 ───
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
    # ── Web-SPA P0: serve the React Workbench at /app (placeholder if unbuilt).
    # Coexists with the legacy dashboard at / until the Phase-1 cutover.
    mount_spa(app)
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
    # /api/schema/erd (HTMX ERD fragment, reusing build_erd_mermaid). Distinct
    # from the snapshot-backed GET /schema view; full Studio layout is S-117.
    app.include_router(schema_router)
    # ── end S-115 ──
    # ── S-118 spec router ──
    # GET /api/spec — DataSpec read endpoint over app.state.workspace (S-111),
    # building a heuristic spec lazily via spec.analyzer.heuristic_fallback
    # when no spec is cached. Content-negotiates: JSON by default, HTMX grid
    # fragment on ``Accept: text/html``. No persistence; no LLM.
    app.include_router(spec_router)
    # ── end S-118 ──
    # ─── S-120 generators region ───
    # GET /api/generators — provider/method catalogue derived from the
    # spec.catalog helpers (heuristic PATTERNS + _TYPE_FALLBACKS). Read-only,
    # workspace-independent; the Studio method-picker fetches this once on
    # open to populate the dropdown.
    app.include_router(generators_router)
    # ─── end S-120 ───
    # ── S-117 studio shell ──
    # GET /studio — single-page workspace shell with four named-slot panels
    # (tree · grid · context · console). Later Phase-C stories (S-118 spec
    # grid, S-125 console progress, S-127 seed control) plug into the stable
    # element ids (#studio-tree / #studio-grid / #studio-context /
    # #studio-console) without editing this shell.
    app.include_router(studio_router)
    # ── end S-117 ──
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
    # POST /api/validate — integrity report over app.state.workspace.last_result.
    # Reuses dbsprout/quality/integrity.py (FK / UNIQUE / NOT NULL); CHECK slot
    # is reserved in the envelope shape for future quality work (S-134+). 409
    # NO_RUN when no generation has run yet. HTMX-aware: returns the
    # ``_validate_panel.html`` fragment with stable ``data-table`` /
    # ``data-column`` / ``data-row`` row attributes so S-135 can wire drill-down.
    app.include_router(validate_router)
    # ── end S-133 ──
    # ── S-142 wizard ──
    # GET /wizard renders the 6-step guided shell (Connect → Review →
    # Configure → Generate → Validate → Insert/Export); GET /wizard/step/{n}
    # returns the HTMX body fragment for step n; POST /wizard/step/{n}
    # persists the submission to app.state.workspace.wizard_state (S-142
    # frozen Pydantic model) and advances / rewinds / jumps. Step bodies are
    # placeholders here — S-143 (Wave 2) wires the real per-step flows
    # without restructuring the shell or the navigation contract.
    app.include_router(wizard_router)
    # ── end S-142 ──
    # ── S-140 export route ──
    # POST /api/export — stream the workspace's last GenerateResult (S-111 / S-124)
    # back to the browser as a single file download. Resolves the writer through
    # dbsprout/plugins/dispatch::resolve_writer (same path the CLI uses) and
    # cleans the per-request temp dir from within the streaming generator. The
    # mount lives inside a delimited region block so S-142 (wizard) can mount
    # alongside without colliding on the same line.
    app.include_router(export_router)
    # ── end S-140 ──
    return app


app = create_app()
