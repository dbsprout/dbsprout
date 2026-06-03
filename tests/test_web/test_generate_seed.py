"""Seed-control + reproducibility tests for ``POST /api/generate`` (S-127).

The companion S-124 ``test_generate.py`` covers the route's skeleton; this
module focuses on S-127's additions:

* ``seed: int | None`` request shape — server fills a non-negative 64-bit int
  when the client says ``null``/omits the field, and **returns the chosen seed**
  in the response so the UI can display it (and let the user copy / re-use it).
* The chosen seed lands on ``JobRecord.seed`` and is surfaced through a new
  ``GET /api/jobs/{job_id}`` JSON envelope.
* End-to-end byte-identical determinism: two ``POST /api/generate`` runs with
  the same explicit seed against the same sqlite fixture produce
  byte-identical output. The assertion is on a stable SHA-256 over the
  serialised ``GenerateResult.tables_data`` to match the AC's "byte-identical
  output" wording without forcing a particular on-disk format.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


# ── fixtures / helpers ─────────────────────────────────────────────────


def _make_app(state_db: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=state_db)


def _temp_sqlite(tmp_path: Path) -> str:
    """Create a 2-table sqlite DB and return its ``sqlite:///`` URL.

    Mirrors ``test_generate.py``'s fixture so the test surfaces stay parallel —
    a future refactor can factor this into a shared conftest helper, but for
    S-127 we keep the local copy so this test file is self-contained.
    """
    db_path = tmp_path / "gen.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute(
            "CREATE TABLE posts (id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id))"
        )
        conn.commit()
    finally:
        conn.close()
    return f"sqlite:///{db_path}"


def _load_schema(app: FastAPI, tmp_path: Path) -> None:
    TestClient(app).post("/api/connect", json={"url": _temp_sqlite(tmp_path)})


def _hash_tables(tables_data: dict[str, list[dict[str, Any]]]) -> str:
    """Deterministically SHA-256 a ``GenerateResult.tables_data`` mapping.

    JSON with ``sort_keys=True`` plus ``default=str`` (for any non-JSON
    primitives such as ``datetime`` / ``Decimal``) gives a stable byte stream
    independent of dict insertion order across runs.
    """
    payload = json.dumps(tables_data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ── request shape: seed is now optional ────────────────────────────────


def test_generate_returns_seed_when_provided(tmp_path: Path) -> None:
    """An explicit integer seed comes back verbatim so the UI can display it."""
    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)
    resp = TestClient(app).post("/api/generate", json={"seed": 12345})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["job_id"], str)
    assert body["seed"] == 12345


def test_generate_materialises_seed_when_seed_null(tmp_path: Path) -> None:
    """``{"seed": null}`` → server picks a non-negative 64-bit int + returns it."""
    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)
    resp = TestClient(app).post("/api/generate", json={"seed": None})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["seed"], int)
    assert 0 <= body["seed"] < (1 << 63)


def test_generate_materialises_seed_when_seed_absent(tmp_path: Path) -> None:
    """Omitting ``seed`` entirely is equivalent to ``"seed": null``."""
    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)
    resp = TestClient(app).post("/api/generate", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["seed"], int)
    assert 0 <= body["seed"] < (1 << 63)


def test_generate_random_seeds_differ_across_calls(tmp_path: Path) -> None:
    """Two random-seed submits should (with overwhelming probability) differ.

    Collision odds at 2**63 over two draws are ~5e-20 — vastly below any test
    flake budget. We compare *one* pair to keep the test fast (the manager is
    single-active so we have to await between submits).
    """
    import anyio  # noqa: PLC0415

    from dbsprout.web.jobs import JobStatus  # noqa: PLC0415

    async def _run() -> tuple[int, int]:
        import httpx  # noqa: PLC0415
        from httpx import ASGITransport  # noqa: PLC0415

        app = _make_app(tmp_path / "state.db")
        _load_schema(app, tmp_path)
        transport = ASGITransport(app=app)
        seeds: list[int] = []
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for _ in range(2):
                resp = await client.post("/api/generate", json={"seed": None})
                assert resp.status_code == 200, resp.text
                body = resp.json()
                seeds.append(body["seed"])
                await app.state.job_manager.wait(body["job_id"])
                record = app.state.job_manager.get(body["job_id"])
                assert record.status is JobStatus.SUCCEEDED, record.error
        return seeds[0], seeds[1]

    a, b = anyio.run(_run)
    assert a != b


# ── JobRecord carries the chosen seed ──────────────────────────────────


@pytest.mark.anyio
async def test_job_record_carries_materialised_seed(tmp_path: Path) -> None:
    """The seed surfaced in the response must equal ``JobRecord.seed``."""
    import httpx  # noqa: PLC0415
    from httpx import ASGITransport  # noqa: PLC0415

    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/generate", json={"seed": None})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    record = app.state.job_manager.get(body["job_id"])
    assert record.seed == body["seed"]


# ── GET /api/jobs/{id} surfaces job metadata ──────────────────────────


def test_get_job_returns_seed_and_engine(tmp_path: Path) -> None:
    """After submit, ``GET /api/jobs/{id}`` returns the seed + engine + status."""
    app = _make_app(tmp_path / "state.db")
    _load_schema(app, tmp_path)
    client = TestClient(app)
    submit = client.post("/api/generate", json={"seed": 7, "engine": "heuristic"})
    assert submit.status_code == 200, submit.text
    job_id = submit.json()["job_id"]

    resp = client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == job_id
    assert body["kind"] == "generate"
    assert body["seed"] == 7
    assert body["engine"] == "heuristic"
    assert body["status"] in {"queued", "running", "succeeded", "failed", "cancelled"}
    assert "started_at" in body
    # ISO-8601-ish — at minimum parseable; we don't assert the exact format
    # to leave room for future precision changes.
    datetime.fromisoformat(body["started_at"].replace("Z", "+00:00"))


def test_get_job_unknown_id_is_404(tmp_path: Path) -> None:
    """Unknown ids return a clean 404 with a string detail (no traceback)."""
    app = _make_app(tmp_path / "state.db")
    resp = TestClient(app).get("/api/jobs/does-not-exist")
    assert resp.status_code == 404
    assert "Traceback" not in resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, str)
    assert detail


# ── End-to-end determinism (AC item) ───────────────────────────────────


@pytest.mark.anyio
async def test_same_seed_yields_byte_identical_output(tmp_path: Path) -> None:
    """End-to-end determinism check: same seed → byte-identical output.

    Drives ``POST /api/generate`` twice with the same explicit seed against the
    same sqlite fixture, joins each job via :meth:`JobManager.wait`, pulls the
    resulting ``tables_data`` off the workspace, and asserts the SHA-256 over a
    deterministic JSON serialisation matches.
    """
    import httpx  # noqa: PLC0415
    from httpx import ASGITransport  # noqa: PLC0415

    from dbsprout.web.jobs import JobStatus  # noqa: PLC0415

    seed = 4242
    hashes: list[str] = []
    for run_idx in range(2):
        app = _make_app(tmp_path / f"state-{run_idx}.db")
        workspace_dir = tmp_path / f"workspace-{run_idx}"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        _load_schema(app, workspace_dir)
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/generate", json={"seed": seed})
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["seed"] == seed
            await app.state.job_manager.wait(body["job_id"])
        record = app.state.job_manager.get(body["job_id"])
        assert record.status is JobStatus.SUCCEEDED, record.error
        result = app.state.workspace.get_last_result()
        assert result is not None
        hashes.append(_hash_tables(result.tables_data))
    assert hashes[0] == hashes[1], f"Output diverged across runs: {hashes!r}"


# The seed input UI (seed field + Generate button + random toggle + "Copy seed"
# affordance) moved to the React SPA in the P1c-5 cutover; the seed *contract*
# (seed echoed in the JSON response, persisted on JobRecord.seed) is covered by
# the ``POST /api/generate`` tests above.
