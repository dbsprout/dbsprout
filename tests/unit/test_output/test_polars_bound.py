"""pyproject.toml polars upper-bound pin (S-094)."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[import-not-found]


def _pyproject_path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            return candidate
    msg = "pyproject.toml not found"
    raise AssertionError(msg)


def test_polars_has_upper_bound() -> None:
    data = tomllib.loads(_pyproject_path().read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    polars_pins = [
        dep for group in ("data", "train") for dep in extras[group] if dep.startswith("polars")
    ]
    assert polars_pins
    for pin in polars_pins:
        assert "<2.0" in pin, f"missing upper bound: {pin}"
