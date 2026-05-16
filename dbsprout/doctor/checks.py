"""Pure environment-health check functions for ``dbsprout doctor``.

Each public check returns a :class:`CheckResult` and performs no console
I/O. ``run_all_checks`` orchestrates every check and never raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

Status = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single environment check."""

    category: str
    name: str
    status: Status
    message: str
    fix: str | None = None


def run_all_checks(
    *,
    db_url: str | None = None,  # noqa: ARG001 - wired up in Task 8
    config_path: Path | None = None,  # noqa: ARG001 - wired up in Task 8
) -> list[CheckResult]:
    """Placeholder — full orchestration implemented in Task 8."""
    return []
