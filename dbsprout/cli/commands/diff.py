"""``dbsprout diff`` command — report schema changes since the last snapshot."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def diff_command(
    db: str | None,
    file: str | None,
    snapshot: str | None,
    output_format: str,
    output_dir: Path,
) -> None:
    """Report schema changes since the last snapshot."""
    raise NotImplementedError
