"""Tests for the dbsprout doctor check engine."""

from __future__ import annotations

import importlib.util
import shutil
from collections import namedtuple
from typing import TYPE_CHECKING

from dbsprout.doctor import CheckResult
from dbsprout.doctor import checks as checks_mod
from dbsprout.doctor.checks import (
    check_database,
    check_extras,
    check_model,
    check_python_version,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_check_result_is_frozen() -> None:
    r = CheckResult(category="Environment", name="python", status="pass", message="ok")
    assert r.fix is None
    try:
        r.status = "fail"  # type: ignore[misc]
    except AttributeError:
        pass
    else:  # pragma: no cover - guard
        raise AssertionError("CheckResult must be frozen")


def test_check_python_version_pass() -> None:
    r = check_python_version((3, 11, 0))
    assert r.status == "pass"
    assert r.category == "Environment"


def test_check_python_version_fail() -> None:
    r = check_python_version((3, 9, 0))
    assert r.status == "fail"
    assert r.fix is not None


def test_check_extras_reports_results() -> None:
    results = check_extras()
    assert results
    assert all(r.category == "Environment" for r in results)
    names = {r.name for r in results}
    assert "extra:sqlalchemy" in names


def test_check_extras_missing_is_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    results = checks_mod.check_extras()
    assert all(r.status == "warn" for r in results)
    assert all(r.fix is not None for r in results)


def test_check_database_no_url_is_pass() -> None:
    r = check_database(None)
    assert r.status == "pass"
    assert "no database" in r.message.lower()


def test_check_database_sqlite_roundtrip(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'd.db'}"
    r = check_database(url)
    assert r.status == "pass"
    assert r.category == "Database"


def test_check_database_bad_url_is_fail() -> None:
    r = check_database("notadialect://x")
    assert r.status == "fail"
    assert r.fix is not None


def test_check_model_absent_is_warn(tmp_path: Path) -> None:
    r = check_model(model_root=tmp_path)
    assert r.status == "warn"
    assert r.fix is not None


def test_check_model_present_is_pass(tmp_path: Path) -> None:
    nested = tmp_path / "models--Qwen" / "snap"
    nested.mkdir(parents=True)
    (nested / "qwen2.5-1.5b-instruct-q4_k_m.gguf").write_bytes(b"x")
    r = check_model(model_root=tmp_path)
    assert r.status == "pass"


def test_check_disk_space_low_is_warn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: usage(100, 100, 1024))
    r = checks_mod.check_disk_space(tmp_path)
    assert r.status == "warn"
