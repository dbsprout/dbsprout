"""Tests for the dbsprout doctor check engine."""

from __future__ import annotations

from dbsprout.doctor import CheckResult
from dbsprout.doctor.checks import check_python_version


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
