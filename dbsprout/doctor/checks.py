"""Pure environment-health check functions for ``dbsprout doctor``.

Each public check returns a :class:`CheckResult` and performs no console
I/O. ``run_all_checks`` orchestrates every check and never raises.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

Status = Literal["pass", "warn", "fail"]

_MIN_PY = (3, 10)

_EXTRA_MODULES: tuple[tuple[str, str], ...] = (
    ("sqlalchemy", "db"),
    ("mimesis", "gen"),
    ("numpy", "gen"),
    ("llama_cpp", "llm"),
    ("huggingface_hub", "llm"),
    ("scipy", "stats"),
    ("sklearn", "stats"),
    ("polars", "data"),
)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single environment check."""

    category: str
    name: str
    status: Status
    message: str
    fix: str | None = None


def check_python_version(
    version_info: tuple[int, int, int] | None = None,
) -> CheckResult:
    """Verify the interpreter is Python 3.10 or newer."""
    vi = version_info if version_info is not None else sys.version_info[:3]
    found = ".".join(str(p) for p in vi)
    if vi[:2] >= _MIN_PY:
        return CheckResult("Environment", "python", "pass", f"Python {found} (>= 3.10)")
    return CheckResult(
        "Environment",
        "python",
        "fail",
        f"Python {found} is too old; 3.10+ required",
        fix="Install Python 3.10 or newer and recreate the virtualenv.",
    )


def _module_version(module: str) -> str:
    dist = {"sklearn": "scikit-learn", "llama_cpp": "llama-cpp-python"}.get(module, module)
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def check_extras() -> list[CheckResult]:
    """Report which optional dependency modules are importable."""
    results: list[CheckResult] = []
    for module, extra in _EXTRA_MODULES:
        present = importlib.util.find_spec(module) is not None
        if present:
            results.append(
                CheckResult(
                    "Environment",
                    f"extra:{module}",
                    "pass",
                    f"{module} {_module_version(module)} installed",
                )
            )
        else:
            results.append(
                CheckResult(
                    "Environment",
                    f"extra:{module}",
                    "warn",
                    f"{module} not installed (optional)",
                    fix=f"Install with: pip install dbsprout[{extra}]",
                )
            )
    return results


def check_database(db_url: str | None) -> CheckResult:
    """Attempt ``SELECT 1`` against a configured database URL."""
    if not db_url:
        return CheckResult(
            "Database",
            "connectivity",
            "pass",
            "No database configured (skipped)",
        )
    import sqlalchemy as sa  # noqa: PLC0415

    from dbsprout.schema.introspect import _create_engine  # noqa: PLC0415

    try:
        safe = sa.engine.make_url(db_url).render_as_string(hide_password=True)
    except Exception:
        safe = "<unparseable url>"
    try:
        engine = _create_engine(db_url)
        try:
            with engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
        finally:
            engine.dispose()
    except Exception as exc:
        return CheckResult(
            "Database",
            "connectivity",
            "fail",
            f"Could not connect to {safe}: {type(exc).__name__}",
            fix="Verify the DB URL/credentials and that the server is reachable.",
        )
    return CheckResult("Database", "connectivity", "pass", f"Connected to {safe}")


def run_all_checks(
    *,
    db_url: str | None = None,  # noqa: ARG001 - wired up in Task 8
    config_path: Path | None = None,  # noqa: ARG001 - wired up in Task 8
) -> list[CheckResult]:
    """Placeholder — full orchestration implemented in Task 8."""
    return []
