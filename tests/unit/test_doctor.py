"""Tests for the dbsprout doctor check engine."""

from __future__ import annotations

from dbsprout.doctor import CheckResult


def test_check_result_is_frozen() -> None:
    r = CheckResult(category="Environment", name="python", status="pass", message="ok")
    assert r.fix is None
    try:
        r.status = "fail"  # type: ignore[misc]
    except AttributeError:
        pass
    else:  # pragma: no cover - guard
        raise AssertionError("CheckResult must be frozen")
