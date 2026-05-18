"""NaN/Inf JSON clamp + Rich markup escape in fidelity output (S-094)."""

from __future__ import annotations

import json
import math

from dbsprout.cli.commands.validate import _print_fidelity_rich
from dbsprout.quality.fidelity import FidelityMetric, FidelityReport
from dbsprout.quality.integrity import IntegrityReport
from dbsprout.quality.report import QualityReport


def test_quality_report_clamps_nan_and_inf_scores() -> None:
    fr = FidelityReport(
        metrics=[FidelityMetric("ks", "t", "c", float("nan"))],
        overall_score=float("inf"),
        passed=False,
    )
    qr = QualityReport.from_reports(
        integrity=IntegrityReport(),
        schema_hash="abc",
        row_counts={"t": 1},
        engine="heuristic",
        seed=42,
        fidelity=fr,
    )
    dumped = qr.model_dump_json()
    assert "NaN" not in dumped
    assert "Infinity" not in dumped
    parsed = json.loads(dumped)  # raises if invalid JSON (e.g. bare NaN)

    # NaN score clamps to a finite 0.0 (not dropped to JSON null); +Inf
    # overall clamps to 1.0. Without the clamp Pydantic emits null here,
    # losing the numeric score downstream consumers expect.
    assert parsed["fidelity"]["metrics"][0]["score"] == 0.0
    assert parsed["fidelity"]["overall_score"] == 1.0
    assert qr.fidelity is not None
    assert math.isfinite(qr.fidelity.metrics[0].score)
    assert qr.fidelity.metrics[0].score == 0.0
    assert qr.fidelity.overall_score == 1.0


def test_fidelity_rich_escapes_markup(capsys) -> None:
    rep = FidelityReport(
        metrics=[FidelityMetric("ks", "[red]evil[/red]", "c", 0.9)],
        overall_score=0.9,
        passed=True,
    )
    _print_fidelity_rich(rep)
    out = capsys.readouterr().out
    assert "[red]evil[/red]" in out
