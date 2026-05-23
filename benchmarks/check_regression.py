"""Benchmark regression gate (S-096).

Compares the current ``benchmark-results.json`` (produced by
``pytest benchmarks/ --benchmark-json=...``) against a committed historical
baseline and fails when any benchmark's mean run time regresses beyond a
threshold (default 20%).

The comparison logic is pure and side-effect free so it can be unit-tested with
synthetic JSON payloads. Only ``main`` performs I/O.

Usage::

    python benchmarks/check_regression.py \\
        --baseline benchmarks/baseline.json \\
        --current benchmark-results.json \\
        [--threshold 0.20]

Exit codes: ``0`` (no regression), ``1`` (one or more regressions), ``2``
(input error, e.g. a missing or malformed JSON file).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

#: Default fractional regression threshold (0.20 == 20% slower).
DEFAULT_THRESHOLD = 0.20


@dataclass(frozen=True)
class Regression:
    """A single benchmark that regressed beyond the threshold."""

    fullname: str
    baseline_mean: float
    current_mean: float
    pct_change: float


def compare_benchmarks(
    baseline: dict[str, float],
    current: dict[str, float],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Regression]:
    """Return the benchmarks whose mean time regressed beyond *threshold*.

    Both inputs map a benchmark ``fullname`` to its mean run time in seconds
    (lower is faster). A benchmark regresses when its current mean exceeds the
    baseline mean by strictly more than ``threshold`` (a fraction, e.g. ``0.20``
    for 20%).

    Benchmarks present in only one of the two inputs are ignored (a new
    benchmark cannot regress; a removed benchmark is not a regression).
    Benchmarks with a non-positive baseline mean are skipped to avoid
    meaningless ratios.
    """
    regressions: list[Regression] = []
    for fullname, base_mean in baseline.items():
        if base_mean <= 0:
            continue
        if fullname not in current:
            continue
        cur_mean = current[fullname]
        pct_change = (cur_mean - base_mean) / base_mean
        if pct_change > threshold:
            regressions.append(
                Regression(
                    fullname=fullname,
                    baseline_mean=base_mean,
                    current_mean=cur_mean,
                    pct_change=pct_change,
                )
            )
    return regressions


def load_benchmarks(path: Path) -> dict[str, float]:
    """Load a pytest-benchmark JSON file into a ``{fullname: mean}`` mapping.

    Mirrors the pytest-benchmark schema: a top-level ``benchmarks`` array whose
    entries each carry a ``fullname`` and a ``stats.mean``. A missing
    ``benchmarks`` key yields an empty mapping.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    benchmarks = raw.get("benchmarks", [])
    means: dict[str, float] = {}
    for entry in benchmarks:
        fullname = entry["fullname"]
        means[fullname] = float(entry["stats"]["mean"])
    return means


def format_report(regressions: list[Regression], threshold: float) -> str:
    """Render a human-readable summary of the comparison result."""
    pct = threshold * 100
    if not regressions:
        return f"No performance regressions detected (threshold: >{pct:.0f}% slower)."
    lines = [f"Performance regressions detected (threshold: >{pct:.0f}% slower):"]
    for reg in sorted(regressions, key=lambda r: r.pct_change, reverse=True):
        lines.append(
            f"  - {reg.fullname}: {reg.baseline_mean:.6g}s -> {reg.current_mean:.6g}s "
            f"({reg.pct_change * 100:+.1f}%)"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="Path to the committed baseline pytest-benchmark JSON.",
    )
    parser.add_argument(
        "--current",
        required=True,
        type=Path,
        help="Path to the current run's pytest-benchmark JSON.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Fractional regression threshold (default: 0.20 == 20%%).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns the process exit code."""
    args = _parse_args(argv)
    try:
        baseline = load_benchmarks(args.baseline)
        current = load_benchmarks(args.current)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: failed to read benchmark JSON: {exc}", file=sys.stderr)
        return 2

    if not baseline:
        print("Warning: baseline contains no benchmarks; skipping regression gate.")
    if not current:
        print("Warning: current run contains no benchmarks; skipping regression gate.")

    regressions = compare_benchmarks(baseline, current, threshold=args.threshold)
    print(format_report(regressions, threshold=args.threshold))
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
