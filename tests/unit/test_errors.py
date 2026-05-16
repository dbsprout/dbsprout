"""Unit tests for the dbsprout error hierarchy (S-076)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover — Python 3.10
    import tomli as tomllib  # type: ignore[import-not-found]

from dbsprout.errors import (
    ConfigError,
    ConnectionError,  # noqa: A004 — intentional user-facing name
    DBSproutError,
    GenerationError,
    MissingDependencyError,
    ModelError,
    SchemaError,
    require_dependency,
)


def test_base_error_carries_what_why_fix() -> None:
    err = DBSproutError(what="It broke", why="Bad input", fix="Try again")
    assert err.what == "It broke"
    assert err.why == "Bad input"
    assert err.fix == "Try again"
    assert err.exit_code == 1


def test_str_is_compact_single_line() -> None:
    err = DBSproutError(what="It broke", why="Bad input", fix="Try again")
    text = str(err)
    assert "\n" not in text
    assert "It broke" in text


def test_subclasses_are_dbsprout_errors() -> None:
    for cls in (ConnectionError, SchemaError, GenerationError, ConfigError, ModelError):
        err = cls(what="w", why="y", fix="f")
        assert isinstance(err, DBSproutError)


def test_custom_exit_code() -> None:
    err = SchemaError(what="w", why="y", fix="f", exit_code=2)
    assert err.exit_code == 2


def test_missing_dependency_builds_extra_command() -> None:
    err = MissingDependencyError(package="polars", extra="data")
    assert "pip install dbsprout[data]" in err.fix
    assert "polars" in err.what
    assert err.exit_code == 1
    assert isinstance(err, DBSproutError)


def test_missing_dependency_no_extra_uses_base_package() -> None:
    err = MissingDependencyError(package="pydbml", extra=None)
    assert "pip install dbsprout" in err.fix
    assert "pip install dbsprout[" not in err.fix


def test_require_dependency_passes_for_present_module() -> None:
    # 'json' is always importable.
    require_dependency("json", extra="data", package="json")


def test_require_dependency_raises_for_missing_module() -> None:
    with pytest.raises(MissingDependencyError) as exc_info:
        require_dependency("definitely_not_a_real_module_xyz", extra="stats")
    assert "pip install dbsprout[stats]" in exc_info.value.fix


_VALID_EXTRAS = {
    "dev",
    "db",
    "mssql",
    "pg",
    "gen",
    "llm",
    "migrate",
    "cloud",
    "privacy",
    "stats",
    "train-cuda",
    "train-mlx",
    "train-dp",
    "web",
    "tui",
    "data",
    "train",
    "mongo",
    "docs",
    "all",
}


def test_all_install_hints_reference_real_extras() -> None:
    root = Path(__file__).resolve().parents[2] / "dbsprout"
    pattern = re.compile(r"pip install dbsprout\[([a-z-]+)\]")
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            if match.group(1) not in _VALID_EXTRAS:
                offenders.append(f"{path}: dbsprout[{match.group(1)}]")
    assert not offenders, f"Invalid extras referenced: {offenders}"


def test_train_dp_extra_declared() -> None:
    root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert "train-dp" in extras
    assert any("opacus" in dep for dep in extras["train-dp"])
