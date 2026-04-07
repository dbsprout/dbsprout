"""Tests for QualityReport Pydantic v2 model and from_reports() factory."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from dbsprout.quality.detection import DetectionMetric, DetectionReport
from dbsprout.quality.fidelity import FidelityMetric, FidelityReport
from dbsprout.quality.integrity import CheckResult, IntegrityReport
from dbsprout.quality.report import (
    IntegrityCheckModel,
    IntegrityReportModel,
    QualityReport,
    ReportMetadata,
)

# ---------------------------------------------------------------------------
# Task 2 tests — Pydantic model structure
# ---------------------------------------------------------------------------


class TestReportMetadata:
    """Tests for ReportMetadata model."""

    def test_report_metadata_fields(self) -> None:
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        meta = ReportMetadata(
            timestamp=ts,
            schema_hash="abc123",
            row_counts={"users": 100, "orders": 50},
            engine="heuristic",
            seed=42,
        )
        assert meta.timestamp == ts
        assert meta.schema_hash == "abc123"
        assert meta.row_counts == {"users": 100, "orders": 50}
        assert meta.engine == "heuristic"
        assert meta.seed == 42


class TestIntegrityCheckModel:
    """Tests for IntegrityCheckModel round-trip."""

    def test_integrity_check_model_round_trip(self) -> None:
        check = IntegrityCheckModel(
            check="pk_unique",
            table="users",
            column="id",
            passed=True,
            details="all unique",
        )
        dumped = check.model_dump()
        restored = IntegrityCheckModel.model_validate(dumped)
        assert restored == check


class TestIntegrityReportModel:
    """Tests for IntegrityReportModel."""

    def test_integrity_report_model_fields(self) -> None:
        checks = [
            IntegrityCheckModel(check="pk", table="t", column="c", passed=True),
            IntegrityCheckModel(check="fk", table="t", column="d", passed=False, details="bad"),
        ]
        report = IntegrityReportModel(passed=False, checks=checks)
        assert report.passed is False
        assert len(report.checks) == 2
        assert report.checks[1].details == "bad"


class TestFidelityOptionalNone:
    """Tests for optional fidelity being None."""

    def test_fidelity_report_model_optional_none(self) -> None:
        meta = ReportMetadata(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            schema_hash="h",
            row_counts={"t": 1},
            engine="e",
            seed=0,
        )
        integrity = IntegrityReportModel(passed=True, checks=[])
        qr = QualityReport(
            metadata=meta,
            integrity=integrity,
            fidelity=None,
            passed=True,
        )
        dumped_json = qr.model_dump_json()
        parsed = json.loads(dumped_json)
        assert "fidelity" in parsed
        assert parsed["fidelity"] is None


class TestDetectionOptionalNone:
    """Tests for optional detection being None."""

    def test_detection_report_model_optional_none(self) -> None:
        meta = ReportMetadata(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            schema_hash="h",
            row_counts={"t": 1},
            engine="e",
            seed=0,
        )
        integrity = IntegrityReportModel(passed=True, checks=[])
        qr = QualityReport(
            metadata=meta,
            integrity=integrity,
            detection=None,
            passed=True,
        )
        dumped_json = qr.model_dump_json()
        parsed = json.loads(dumped_json)
        assert "detection" in parsed
        assert parsed["detection"] is None


class TestQualityReportVersionField:
    """Tests for version field default."""

    def test_quality_report_version_field(self) -> None:
        meta = ReportMetadata(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            schema_hash="h",
            row_counts={},
            engine="e",
            seed=0,
        )
        integrity = IntegrityReportModel(passed=True, checks=[])
        qr = QualityReport(metadata=meta, integrity=integrity, passed=True)
        assert qr.version == "1.0.0"


class TestQualityReportJson:
    """Tests for JSON serialization."""

    def test_quality_report_model_dump_json_valid(self) -> None:
        meta = ReportMetadata(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            schema_hash="h",
            row_counts={"t": 10},
            engine="e",
            seed=1,
        )
        integrity = IntegrityReportModel(passed=True, checks=[])
        qr = QualityReport(metadata=meta, integrity=integrity, passed=True)
        result = qr.model_dump_json(indent=2)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert parsed["version"] == "1.0.0"

    def test_quality_report_compact_json(self) -> None:
        meta = ReportMetadata(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            schema_hash="h",
            row_counts={},
            engine="e",
            seed=0,
        )
        integrity = IntegrityReportModel(passed=True, checks=[])
        qr = QualityReport(metadata=meta, integrity=integrity, passed=True)
        compact = qr.model_dump_json()
        assert "  " not in compact


class TestQualityReportFrozen:
    """Tests for frozen (immutable) config."""

    def test_quality_report_frozen(self) -> None:
        meta = ReportMetadata(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            schema_hash="h",
            row_counts={},
            engine="e",
            seed=0,
        )
        integrity = IntegrityReportModel(passed=True, checks=[])
        qr = QualityReport(metadata=meta, integrity=integrity, passed=True)
        with pytest.raises(ValidationError):
            qr.version = "2.0.0"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Task 3 tests — from_reports() factory method
# ---------------------------------------------------------------------------


class TestFromReportsIntegrityOnly:
    """Tests for from_reports with integrity only."""

    def test_from_reports_integrity_only(self) -> None:
        ir = IntegrityReport(
            checks=[CheckResult(check="pk", table="t", column="c", passed=True)],
            passed=True,
        )
        qr = QualityReport.from_reports(
            integrity=ir,
            schema_hash="abc",
            row_counts={"t": 10},
            engine="heuristic",
            seed=42,
        )
        assert qr.integrity.passed is True
        assert len(qr.integrity.checks) == 1
        assert qr.integrity.checks[0].check == "pk"
        assert qr.fidelity is None
        assert qr.detection is None
        assert qr.passed is True
        assert qr.version == "1.0.0"


class TestFromReportsAllThree:
    """Tests for from_reports with all three sub-reports."""

    def test_from_reports_all_three(self) -> None:
        ir = IntegrityReport(
            checks=[CheckResult(check="pk", table="t", column="c", passed=True)],
            passed=True,
        )
        fr = FidelityReport(
            metrics=[
                FidelityMetric(
                    metric="ks",
                    table="t",
                    column="c",
                    score=0.95,
                    details="good",
                )
            ],
            overall_score=0.95,
            passed=True,
        )
        dr = DetectionReport(
            metrics=[
                DetectionMetric(
                    metric="lr",
                    table="t",
                    accuracy=0.52,
                    details="ok",
                )
            ],
            overall_score=0.52,
            passed=True,
        )
        qr = QualityReport.from_reports(
            integrity=ir,
            schema_hash="abc",
            row_counts={"t": 10},
            engine="heuristic",
            seed=42,
            fidelity=fr,
            detection=dr,
        )
        assert qr.integrity.passed is True
        assert qr.fidelity is not None
        assert qr.fidelity.overall_score == 0.95
        assert len(qr.fidelity.metrics) == 1
        assert qr.detection is not None
        assert qr.detection.overall_score == 0.52
        assert len(qr.detection.metrics) == 1
        assert qr.passed is True


class TestFromReportsIntegrityFails:
    """Tests for from_reports when integrity fails."""

    def test_from_reports_integrity_fails(self) -> None:
        ir = IntegrityReport(
            checks=[CheckResult(check="pk", table="t", column="c", passed=False)],
            passed=False,
        )
        qr = QualityReport.from_reports(
            integrity=ir,
            schema_hash="abc",
            row_counts={"t": 10},
            engine="heuristic",
            seed=42,
        )
        assert qr.passed is False


class TestFromReportsFidelityFails:
    """Tests for from_reports when fidelity fails."""

    def test_from_reports_fidelity_fails(self) -> None:
        ir = IntegrityReport(checks=[], passed=True)
        fr = FidelityReport(metrics=[], overall_score=0.3, passed=False)
        qr = QualityReport.from_reports(
            integrity=ir,
            schema_hash="abc",
            row_counts={"t": 10},
            engine="heuristic",
            seed=42,
            fidelity=fr,
        )
        assert qr.passed is False


class TestFromReportsDetectionFails:
    """Tests for from_reports when detection fails."""

    def test_from_reports_detection_fails(self) -> None:
        ir = IntegrityReport(checks=[], passed=True)
        dr = DetectionReport(metrics=[], overall_score=0.9, passed=False)
        qr = QualityReport.from_reports(
            integrity=ir,
            schema_hash="abc",
            row_counts={"t": 10},
            engine="heuristic",
            seed=42,
            detection=dr,
        )
        assert qr.passed is False


class TestFromReportsMetadataTimestamp:
    """Tests for metadata timestamp being recent."""

    def test_from_reports_metadata_timestamp_recent(self) -> None:
        ir = IntegrityReport(checks=[], passed=True)
        qr = QualityReport.from_reports(
            integrity=ir,
            schema_hash="abc",
            row_counts={"t": 10},
            engine="heuristic",
            seed=42,
        )
        now = datetime.now(timezone.utc)
        delta = (now - qr.metadata.timestamp).total_seconds()
        assert delta < 5.0
