"""Quality report Pydantic v2 model — JSON-serializable wrapper.

Provides frozen Pydantic models that mirror the existing frozen dataclass
reports (IntegrityReport, FidelityReport, DetectionReport) and a
QualityReport envelope with a ``from_reports()`` factory method.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from dbsprout.quality.detection import DetectionReport
    from dbsprout.quality.fidelity import FidelityReport
    from dbsprout.quality.integrity import IntegrityReport


class ReportMetadata(BaseModel):
    """Metadata about the quality report run."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    schema_hash: str
    row_counts: dict[str, int]
    engine: str
    seed: int


class IntegrityCheckModel(BaseModel):
    """Pydantic mirror of ``CheckResult`` dataclass."""

    model_config = ConfigDict(frozen=True)

    check: str
    table: str
    column: str
    passed: bool
    details: str = ""


class IntegrityReportModel(BaseModel):
    """Pydantic mirror of ``IntegrityReport`` dataclass."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    checks: list[IntegrityCheckModel]


class FidelityMetricModel(BaseModel):
    """Pydantic mirror of ``FidelityMetric`` dataclass."""

    model_config = ConfigDict(frozen=True)

    metric: str
    table: str
    column: str
    score: float
    details: str = ""


class FidelityReportModel(BaseModel):
    """Pydantic mirror of ``FidelityReport`` dataclass."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    overall_score: float
    metrics: list[FidelityMetricModel]


class DetectionMetricModel(BaseModel):
    """Pydantic mirror of ``DetectionMetric`` dataclass."""

    model_config = ConfigDict(frozen=True)

    metric: str
    table: str
    accuracy: float
    details: str = ""


class DetectionReportModel(BaseModel):
    """Pydantic mirror of ``DetectionReport`` dataclass."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    overall_score: float
    metrics: list[DetectionMetricModel]


class QualityReport(BaseModel):
    """Top-level quality report envelope — JSON-serializable."""

    model_config = ConfigDict(frozen=True)

    version: str = "1.0.0"
    metadata: ReportMetadata
    integrity: IntegrityReportModel
    fidelity: FidelityReportModel | None = None
    detection: DetectionReportModel | None = None
    passed: bool

    @classmethod
    def from_reports(  # noqa: PLR0913
        cls,
        integrity: IntegrityReport,
        schema_hash: str,
        row_counts: dict[str, int],
        engine: str,
        seed: int,
        fidelity: FidelityReport | None = None,
        detection: DetectionReport | None = None,
    ) -> QualityReport:
        """Build a ``QualityReport`` from frozen dataclass sub-reports."""
        integrity_model = IntegrityReportModel(
            passed=integrity.passed,
            checks=[
                IntegrityCheckModel(
                    check=c.check,
                    table=c.table,
                    column=c.column,
                    passed=c.passed,
                    details=c.details,
                )
                for c in integrity.checks
            ],
        )

        fidelity_model: FidelityReportModel | None = None
        if fidelity is not None:
            fidelity_model = FidelityReportModel(
                passed=fidelity.passed,
                overall_score=fidelity.overall_score,
                metrics=[
                    FidelityMetricModel(
                        metric=m.metric,
                        table=m.table,
                        column=m.column,
                        score=m.score,
                        details=m.details,
                    )
                    for m in fidelity.metrics
                ],
            )

        detection_model: DetectionReportModel | None = None
        if detection is not None:
            detection_model = DetectionReportModel(
                passed=detection.passed,
                overall_score=detection.overall_score,
                metrics=[
                    DetectionMetricModel(
                        metric=m.metric,
                        table=m.table,
                        accuracy=m.accuracy,
                        details=m.details,
                    )
                    for m in detection.metrics
                ],
            )

        passed = integrity.passed
        if fidelity is not None:
            passed = passed and fidelity.passed
        if detection is not None:
            passed = passed and detection.passed

        metadata = ReportMetadata(
            timestamp=datetime.now(timezone.utc),
            schema_hash=schema_hash,
            row_counts=row_counts,
            engine=engine,
            seed=seed,
        )

        return cls(
            metadata=metadata,
            integrity=integrity_model,
            fidelity=fidelity_model,
            detection=detection_model,
            passed=passed,
        )
