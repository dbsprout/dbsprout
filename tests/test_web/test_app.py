"""FastAPI web app tests (S-090; P1c-5 cutover).

The web stack lives in the optional ``[web]`` extra, so every test guards with
``pytest.importorskip("fastapi")`` *before* importing FastAPI symbols. A
subprocess probe verifies the CLI lazy-import contract — importing
``dbsprout.cli.app`` must never pull FastAPI/uvicorn (mirrors the textual probe
in ``tests/test_tui/test_app.py``). Help-text assertions use ``_strip_ansi`` plus
``COLUMNS``/``NO_COLOR`` env so they pass in CI's TTY-less environment.

Since the P1c-5 cutover the server exposes only the SPA at ``/app`` and the JSON
``/api/*`` API (plus the progress WebSocket). ``GET /`` redirects to ``/app``;
the legacy server-rendered pages (``/``, ``/wizard*``, ``/studio``, ``/quality``,
``/preview``, ``/costs``, ``/history``, ``/schema``, ``/progress``) and the
``POST /api/update-column`` endpoint are gone (404), as is the ``/static`` mount.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path

_ENV = {"COLUMNS": "200", "NO_COLOR": "1"}


def _strip_ansi(text: str) -> str:
    return re.compile(r"\x1b\[[0-9;]*m").sub("", text)


def _make_client(state_db: Path) -> TestClient:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    app = create_app(state_db_path=state_db)
    return TestClient(app)


# ── `/` redirects to the SPA (P1c-5) ──────────────────────────────────


def test_root_redirects_to_app(tmp_path: Path) -> None:
    """``GET /`` is a 308 permanent redirect to the SPA front door ``/app``."""
    resp = _make_client(tmp_path / "state.db").get("/", follow_redirects=False)
    assert resp.status_code == 308
    assert resp.headers["location"] == "/app"


def test_app_serves_spa(tmp_path: Path) -> None:
    """``/app`` serves the SPA (the placeholder page in a Python-only checkout)."""
    resp = _make_client(tmp_path / "state.db").get("/app")
    assert resp.status_code == 200
    assert "DBSprout Workbench" in resp.text


# ── legacy server-rendered surfaces are gone (404) ────────────────────


@pytest.mark.parametrize(
    "path",
    [
        "/wizard",
        "/wizard/step/1",
        "/studio",
        "/quality",
        "/preview",
        "/costs",
        "/history",
        "/schema",
        "/progress",
        "/progress/stream",
        "/static/style.css",
    ],
)
def test_legacy_get_routes_return_404(tmp_path: Path, path: str) -> None:
    resp = _make_client(tmp_path / "state.db").get(path)
    assert resp.status_code == 404, f"{path} should be gone after the P1c-5 cutover"


def test_legacy_update_column_route_returns_404(tmp_path: Path) -> None:
    """``POST /api/update-column`` was removed (superseded by PUT spec column)."""
    resp = _make_client(tmp_path / "state.db").post(
        "/api/update-column", json={"table": "t", "column": "c"}
    )
    assert resp.status_code == 404


# ── JSON API + WebSocket are intact ───────────────────────────────────


def test_health_returns_ok_json(tmp_path: Path) -> None:
    resp = _make_client(tmp_path / "state.db").get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_representative_json_route_returns_json(tmp_path: Path) -> None:
    """A data route returns JSON (no HTML branch) — `/api/spec` with no schema."""
    resp = _make_client(tmp_path / "state.db").get("/api/spec")
    assert resp.status_code == 409
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json()["detail"]["code"] == "NO_SCHEMA"


def test_progress_ws_route_is_registered(tmp_path: Path) -> None:
    """The progress WebSocket survives the cutover (only the SSE page went)."""
    app = _make_client(tmp_path / "state.db").app
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/ws/jobs/{job_id}" in paths
    assert "/app" in paths


def test_no_static_mount(tmp_path: Path) -> None:
    """The legacy ``/static`` mount was removed with the Jinja2/HTMX assets."""
    app = _make_client(tmp_path / "state.db").app
    names = {getattr(r, "name", None) for r in app.routes}
    assert "static" not in names


# ── extension seam contract ───────────────────────────────────────────


def test_router_is_importable_and_extensible() -> None:
    """The shared router still exists (now carries only /health)."""
    from fastapi import APIRouter  # noqa: PLC0415

    from dbsprout.web.routes import router  # noqa: PLC0415

    assert isinstance(router, APIRouter)


def test_state_db_path_honors_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_db = tmp_path / "env.db"
    monkeypatch.setenv("DBSPROUT_STATE_DB", str(state_db))
    from dbsprout.web.app import create_app  # noqa: PLC0415

    client = TestClient(create_app())
    # The env-resolved DB is used: the runs JSON endpoint reads it without error.
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    assert resp.json()["total_runs"] == 0


# ── CLI `serve` command ───────────────────────────────────────────────


def test_serve_invokes_uvicorn(tmp_path: Path) -> None:
    import dbsprout.cli.serve as serve_mod  # noqa: PLC0415

    calls: list[dict[str, object]] = []

    def _fake_run(app_target: object, **kwargs: object) -> None:
        calls.append({"app": app_target, **kwargs})

    fake_uvicorn = type("U", (), {"run": staticmethod(_fake_run)})()
    import sys as _sys  # noqa: PLC0415

    _sys.modules["uvicorn"] = fake_uvicorn  # type: ignore[assignment]
    try:
        serve_mod.serve_command(host="127.0.0.1", port=8420, reload=False)
    finally:
        del _sys.modules["uvicorn"]

    assert len(calls) == 1
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["port"] == 8420


def test_serve_command_listed_in_help() -> None:
    from typer.testing import CliRunner  # noqa: PLC0415

    from dbsprout.cli.app import app  # noqa: PLC0415

    result = CliRunner().invoke(app, ["--help"], env=_ENV)
    assert result.exit_code == 0, result.output
    assert "serve" in _strip_ansi(result.output)


def test_serve_proxy_default_host_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner  # noqa: PLC0415

    import dbsprout.cli.serve as serve_mod  # noqa: PLC0415
    from dbsprout.cli.app import app  # noqa: PLC0415

    captured: dict[str, object] = {}

    def _fake_serve(*, host: str, port: int, reload: bool) -> None:
        captured.update(host=host, port=port, reload=reload)

    monkeypatch.setattr(serve_mod, "serve_command", _fake_serve)
    result = CliRunner().invoke(app, ["serve"], env=_ENV)
    assert result.exit_code == 0, result.output
    assert captured == {"host": "127.0.0.1", "port": 8420, "reload": False}


def test_serve_raises_clear_error_when_fastapi_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core-only install (no [web]) → actionable MissingDependencyError."""
    import builtins  # noqa: PLC0415

    import dbsprout.cli.serve as serve_mod  # noqa: PLC0415
    from dbsprout.errors import MissingDependencyError  # noqa: PLC0415

    real_import = builtins.__import__

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "uvicorn" or name.startswith("uvicorn."):
            raise ImportError("no uvicorn")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(MissingDependencyError) as exc:
        serve_mod.serve_command(host="127.0.0.1", port=8420, reload=False)
    assert exc.value.extra == "web"


# ── lazy-import contract ──────────────────────────────────────────────


def test_cli_app_does_not_import_fastapi_eagerly() -> None:
    """Importing the CLI must not pull FastAPI/uvicorn (preserves <500ms startup)."""
    probe = (
        "import sys\n"
        "import dbsprout.cli.app  # noqa: F401\n"
        "print('fastapi' in sys.modules or 'uvicorn' in sys.modules)\n"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, trusted interpreter
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout.strip() == "False", (
        "importing dbsprout.cli.app eagerly imported fastapi/uvicorn; the `serve` "
        "proxy must lazy-import web deps inside the command body."
    )
