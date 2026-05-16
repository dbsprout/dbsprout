"""Tests for the dbsprout doctor check engine."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

from dbsprout.doctor import CheckResult
from dbsprout.doctor import checks as checks_mod
from dbsprout.doctor.checks import check_extras, check_python_version

if TYPE_CHECKING:
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
