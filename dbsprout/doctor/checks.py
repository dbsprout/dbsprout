"""Pure environment-health check functions for ``dbsprout doctor``.

Each public check returns a :class:`CheckResult` and performs no console
I/O. ``run_all_checks`` orchestrates every check and never raises.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

Status = Literal["pass", "warn", "fail"]

_MIN_PY = (3, 10)


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


def run_all_checks(
    *,
    db_url: str | None = None,  # noqa: ARG001 - wired up in Task 8
    config_path: Path | None = None,  # noqa: ARG001 - wired up in Task 8
) -> list[CheckResult]:
    """Placeholder — full orchestration implemented in Task 8."""
    return []
