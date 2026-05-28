"""Unit tests for the benchmark regression gate (S-096).

These tests exercise the pure comparison logic and the JSON/CLI plumbing in
``benchmarks/check_regression.py`` against synthetic pytest-benchmark JSON
payloads. They are deterministic and fast: no real benchmarks are run.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from benchmarks.check_regression import (
    Regression,
    compare_benchmarks,
    format_report,
    load_benchmarks,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path


def _bench(fullname: str, mean: float) -> dict[str, Any]:
    """Build a minimal benchmark entry mirroring the pytest-benchmark schema."""
    return {
        "name": fullname.rsplit("::", maxsplit=1)[-1],
        "fullname": fullname,
        "stats": {"mean": mean, "median": mean, "min": mean, "ops": 1.0 / mean},
    }


def _report(benchmarks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"version": "5.2.3", "benchmarks": benchmarks}


# ─── compare_benchmarks: core pass/fail logic ──────────────────────────────


def test_no_regression_when_within_threshold() -> None:
    baseline = {"b::t": 1.0}
    current = {"b::t": 1.10}  # 10% slower, under 20%
    assert compare_benchmarks(baseline, current, threshold=0.20) == []


def test_regression_detected_above_threshold() -> None:
    baseline = {"b::t": 1.0}
    current = {"b::t": 1.25}  # 25% slower, over 20%
    regressions = compare_benchmarks(baseline, current, threshold=0.20)
    assert len(regressions) == 1
    reg = regressions[0]
    assert isinstance(reg, Regression)
    assert reg.fullname == "b::t"
    assert reg.baseline_mean == pytest.approx(1.0)
    assert reg.current_mean == pytest.approx(1.25)
    assert reg.pct_change == pytest.approx(0.25)


def test_exactly_at_threshold_is_not_a_regression() -> None:
    baseline = {"b::t": 1.0}
    current = {"b::t": 1.20}  # exactly 20% — strict greater-than, not a regression
    assert compare_benchmarks(baseline, current, threshold=0.20) == []


def test_faster_is_never_a_regression() -> None:
    baseline = {"b::t": 1.0}
    current = {"b::t": 0.5}  # 2x faster
    assert compare_benchmarks(baseline, current, threshold=0.20) == []


def test_new_benchmark_only_in_current_is_ignored() -> None:
    baseline = {"b::a": 1.0}
    current = {"b::a": 1.0, "b::new": 99.0}
    assert compare_benchmarks(baseline, current, threshold=0.20) == []


def test_removed_benchmark_only_in_baseline_is_ignored() -> None:
    baseline = {"b::a": 1.0, "b::gone": 1.0}
    current = {"b::a": 1.0}
    assert compare_benchmarks(baseline, current, threshold=0.20) == []


def test_nonpositive_baseline_mean_is_skipped() -> None:
    baseline = {"b::zero": 0.0, "b::neg": -1.0}
    current = {"b::zero": 5.0, "b::neg": 5.0}
    assert compare_benchmarks(baseline, current, threshold=0.20) == []


def test_empty_inputs_yield_no_regressions() -> None:
    assert compare_benchmarks({}, {}, threshold=0.20) == []


def test_multiple_benchmarks_only_regressed_reported() -> None:
    baseline = {"b::ok": 1.0, "b::bad": 1.0, "b::also_bad": 2.0}
    current = {"b::ok": 1.05, "b::bad": 1.5, "b::also_bad": 3.0}
    regressions = compare_benchmarks(baseline, current, threshold=0.20)
    names = sorted(r.fullname for r in regressions)
    assert names == ["b::also_bad", "b::bad"]


def test_custom_threshold_is_respected() -> None:
    baseline = {"b::t": 1.0}
    current = {"b::t": 1.05}  # 5% slower
    assert compare_benchmarks(baseline, current, threshold=0.20) == []
    assert len(compare_benchmarks(baseline, current, threshold=0.01)) == 1


# ─── load_benchmarks: JSON parsing ─────────────────────────────────────────


def test_load_benchmarks_keys_by_fullname(tmp_path: Path) -> None:
    payload = _report([_bench("benchmarks/bench_cli.py::test_x", 0.25)])
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_benchmarks(path)
    assert loaded == {"benchmarks/bench_cli.py::test_x": pytest.approx(0.25)}


def test_load_benchmarks_empty_array(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(_report([])), encoding="utf-8")
    assert load_benchmarks(path) == {}


def test_load_benchmarks_missing_benchmarks_key(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": "5.2.3"}), encoding="utf-8")
    assert load_benchmarks(path) == {}


# ─── format_report: human-readable output ──────────────────────────────────


def test_format_report_no_regressions() -> None:
    text = format_report([], threshold=0.20)
    assert "No performance regressions" in text


def test_format_report_lists_each_regression() -> None:
    regressions = [
        Regression(fullname="b::slow", baseline_mean=1.0, current_mean=1.5, pct_change=0.5),
    ]
    text = format_report(regressions, threshold=0.20)
    assert "b::slow" in text
    assert "50" in text  # percentage shown


# ─── main: CLI entrypoint + exit codes ─────────────────────────────────────


def _write_pair(tmp_path: Path, base_mean: float, cur_mean: float) -> tuple[Path, Path]:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(json.dumps(_report([_bench("b::t", base_mean)])), encoding="utf-8")
    current.write_text(json.dumps(_report([_bench("b::t", cur_mean)])), encoding="utf-8")
    return baseline, current


def test_main_passes_when_no_regression(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    baseline, current = _write_pair(tmp_path, 1.0, 1.1)
    code = main(["--baseline", str(baseline), "--current", str(current)])
    assert code == 0
    assert "No performance regressions" in capsys.readouterr().out


def test_main_fails_on_regression(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    baseline, current = _write_pair(tmp_path, 1.0, 1.5)
    code = main(["--baseline", str(baseline), "--current", str(current)])
    assert code == 1
    assert "b::t" in capsys.readouterr().out


def test_main_respects_threshold_flag(tmp_path: Path) -> None:
    baseline, current = _write_pair(tmp_path, 1.0, 1.05)
    assert main(["--baseline", str(baseline), "--current", str(current)]) == 0
    assert (
        main(["--baseline", str(baseline), "--current", str(current), "--threshold", "0.01"]) == 1
    )


def test_main_returns_input_error_for_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline, _ = _write_pair(tmp_path, 1.0, 1.0)
    code = main(["--baseline", str(baseline), "--current", str(tmp_path / "nope.json")])
    assert code == 2
    assert "Error" in capsys.readouterr().err
