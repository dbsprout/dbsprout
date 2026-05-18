"""IntegrityReport/FidelityReport true immutability via tuple (S-094)."""

from __future__ import annotations

from dbsprout.quality.fidelity import FidelityMetric, FidelityReport
from dbsprout.quality.integrity import CheckResult, IntegrityReport


def test_integrity_report_checks_default_is_tuple() -> None:
    assert isinstance(IntegrityReport().checks, tuple)


def test_fidelity_report_metrics_default_is_tuple() -> None:
    assert isinstance(FidelityReport().metrics, tuple)


def test_integrity_report_checks_preserved_as_tuple() -> None:
    check = CheckResult(check="pk", table="t", column="id", passed=True)
    rep = IntegrityReport(checks=(check,), passed=True)
    assert isinstance(rep.checks, tuple)
    assert rep.checks[0].column == "id"


def test_fidelity_report_metrics_preserved_as_tuple() -> None:
    metric = FidelityMetric(metric="ks", table="t", column="c", score=0.9)
    rep = FidelityReport(metrics=(metric,), overall_score=0.9, passed=True)
    assert isinstance(rep.metrics, tuple)
    assert rep.metrics[0].score == 0.9
