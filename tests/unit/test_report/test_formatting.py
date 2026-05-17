"""Unit tests for duration formatting + generator config edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from dbsprout.report import DEFAULT_OUTPUT_PATH, ReportGenerator
from dbsprout.report.context import _format_duration


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("ms", "expected"),
        [
            (None, "—"),
            (0, "0 ms"),
            (250, "250 ms"),
            (1500, "1.50 s"),
            (59_000, "59.00 s"),
            (65_000, "1m 5s"),
            (3_661_000, "61m 1s"),
        ],
    )
    def test_branches(self, ms: int | None, expected: str) -> None:
        assert _format_duration(ms) == expected


class TestGeneratorConfig:
    def test_default_output_path_property(self) -> None:
        gen = ReportGenerator()
        assert gen.output_path == DEFAULT_OUTPUT_PATH
        assert gen.output_path == Path("seeds") / "report.html"

    def test_custom_output_path_property(self, tmp_path: Path) -> None:
        target = tmp_path / "x" / "out.html"
        gen = ReportGenerator(output_path=target)
        assert gen.output_path == target

    def test_str_output_path_accepted(self) -> None:
        gen = ReportGenerator(output_path="custom/report.html")
        assert gen.output_path == Path("custom/report.html")
