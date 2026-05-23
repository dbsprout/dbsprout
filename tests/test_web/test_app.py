"""FastAPI web dashboard skeleton tests (S-090).

The web stack lives in the optional ``[web]`` extra, so every test guards with
``pytest.importorskip("fastapi")`` *before* importing FastAPI symbols. A
subprocess probe verifies the CLI lazy-import contract — importing
``dbsprout.cli.app`` must never pull FastAPI/uvicorn (mirrors the textual probe
in ``tests/test_tui/test_app.py``). Help-text assertions use ``_strip_ansi`` plus
``COLUMNS``/``NO_COLOR`` env so they pass in CI's TTY-less environment.
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.state.db import StateDB
from dbsprout.state.models import RunRecord, TableStats

if TYPE_CHECKING:
    from pathlib import Path

_ENV = {"COLUMNS": "200", "NO_COLOR": "1"}


def _strip_ansi(text: str) -> str:
    return re.compile(r"\x1b\[[0-9;]*m").sub("", text)


def _make_client(state_db: Path) -> TestClient:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    app = create_app(state_db_path=state_db)
    return TestClient(app)


def _seed_run(state_db: Path) -> None:
    db = StateDB(state_db)
    db.record_run(
        RunRecord(
            started_at=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
            completed_at=datetime(2026, 5, 20, 12, 0, 5, tzinfo=timezone.utc),
            duration_ms=5000,
            engine="heuristic",
            total_rows=4242,
            total_tables=3,
            seed=42,
            table_stats=[TableStats(table_name="users", row_count=100)],
        )
    )


# ── home dashboard ──────────────────────────────────────────────────


def test_index_returns_200_and_nav(tmp_path: Path) -> None:
    client = _make_client(tmp_path / "state.db")
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    for nav_id in ('id="nav-home"', 'id="nav-schema"', 'id="nav-progress"', 'id="nav-quality"'):
        assert nav_id in body, f"missing stable nav id {nav_id} (sibling extension seam)"


def test_index_loads_cdn_assets_no_build_step(tmp_path: Path) -> None:
    body = _make_client(tmp_path / "state.db").get("/").text
    assert "daisyui" in body.lower(), "DaisyUI must load via CDN"
    assert "htmx" in body.lower(), "HTMX must load via CDN"
    assert "cdn.tailwindcss.com" in body or "tailwind" in body.lower()


def test_index_shows_no_runs_when_state_empty(tmp_path: Path) -> None:
    body = _make_client(tmp_path / "state.db").get("/").text
    assert "No runs yet" in body


def test_index_shows_runs_when_state_has_runs(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _seed_run(state_db)
    body = _make_client(state_db).get("/").text
    assert "heuristic" in body
    assert "4,242" in body or "4242" in body


def test_index_works_when_state_db_missing(tmp_path: Path) -> None:
    """AC: dashboard works even when the CLI has never run (no state.db yet)."""
    resp = _make_client(tmp_path / "never_created.db").get("/")
    assert resp.status_code == 200


# ── health probe ──────────────────────────────────────────────────────


def test_health_returns_ok_json(tmp_path: Path) -> None:
    resp = _make_client(tmp_path / "state.db").get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── sibling-view placeholder tabs (extension seam) ────────────────────


@pytest.mark.parametrize("path", ["/schema", "/progress", "/quality"])
def test_placeholder_tabs_return_200(tmp_path: Path, path: str) -> None:
    resp = _make_client(tmp_path / "state.db").get(path)
    assert resp.status_code == 200
    assert "Coming soon" in resp.text


# ── static assets ─────────────────────────────────────────────────────


def test_static_css_served(tmp_path: Path) -> None:
    resp = _make_client(tmp_path / "state.db").get("/static/style.css")
    assert resp.status_code == 200
    assert "css" in resp.headers["content-type"]


# ── extension seam contract ───────────────────────────────────────────


def test_router_is_importable_and_extensible() -> None:
    """Siblings (S-091/092/093) mount routes onto this shared router."""
    from fastapi import APIRouter  # noqa: PLC0415

    from dbsprout.web.routes import router  # noqa: PLC0415

    assert isinstance(router, APIRouter)


def test_state_db_path_honors_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_db = tmp_path / "env.db"
    _seed_run(state_db)
    monkeypatch.setenv("DBSPROUT_STATE_DB", str(state_db))
    from dbsprout.web.app import create_app  # noqa: PLC0415

    client = TestClient(create_app())
    assert "heuristic" in client.get("/").text


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
