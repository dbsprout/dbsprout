"""Tests for the Workbench SPA mount (P0 foundation).

The web stack is the optional ``[web]`` extra, so we guard with
``pytest.importorskip("fastapi")`` like the sibling web tests. These tests need
NO Node build: the "built" case writes a fake ``index.html`` into a temp dir.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from dbsprout.web.spa import mount_spa, spa_is_built

if TYPE_CHECKING:
    from pathlib import Path


def test_spa_is_built_false_for_empty_dir(tmp_path: Path) -> None:
    assert spa_is_built(tmp_path) is False


def test_spa_is_built_true_when_index_present(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<h1>built</h1>", encoding="utf-8")
    assert spa_is_built(tmp_path) is True


def test_mount_serves_placeholder_when_unbuilt(tmp_path: Path) -> None:
    app = FastAPI()
    mount_spa(app, tmp_path / "missing")
    resp = TestClient(app).get("/app")
    assert resp.status_code == 200
    assert "front-end build is not present" in resp.text


def test_mount_serves_index_when_built(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<h1>built</h1>", encoding="utf-8")
    app = FastAPI()
    mount_spa(app, tmp_path)
    resp = TestClient(app).get("/app/")
    assert resp.status_code == 200
    assert "<h1>built</h1>" in resp.text


def test_create_app_serves_app_route_and_keeps_legacy_home(tmp_path: Path) -> None:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    app = create_app(state_db_path=tmp_path / "state.db")
    client = TestClient(app)
    assert client.get("/app").status_code == 200  # SPA (real or placeholder)
    assert client.get("/").status_code == 200  # legacy dashboard still live
