"""pyproject.toml polars upper-bound pin (S-094)."""

from __future__ import annotations

import sys
from pathlib import Path

import tomllib


def _pyproject_path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            return candidate
    msg = "pyproject.toml not found"
    raise AssertionError(msg)


def test_polars_has_upper_bound() -> None:
    assert sys.version_info >= (3, 11)
    data = tomllib.loads(_pyproject_path().read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    polars_pins = [
        dep for group in ("data", "train") for dep in extras[group] if dep.startswith("polars")
    ]
    assert polars_pins
    for pin in polars_pins:
        assert "<2.0" in pin, f"missing upper bound: {pin}"
