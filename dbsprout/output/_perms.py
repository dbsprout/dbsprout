"""Shared output-file permission hardening.

Restricts generated data files to owner read/write + group read (``0o640``)
instead of the world-readable default ``0o644``. POSIX-only; a graceful no-op
on platforms (e.g. Windows) without POSIX permission bits or for files that do
not exist.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_RESTRICTED_MODE = 0o640


def restrict_file_permissions(path: Path) -> None:
    """Restrict *path* to ``0o640``. No-op off POSIX or if absent."""
    if os.name != "posix":
        return
    with contextlib.suppress(OSError):
        os.chmod(path, _RESTRICTED_MODE)
