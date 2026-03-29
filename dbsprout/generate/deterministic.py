"""Deterministic per-column seed derivation.

Uses SHA-256 (not Python ``hash()``) so seeds are stable across runs,
Python versions, and platforms regardless of ``PYTHONHASHSEED``.
"""

from __future__ import annotations

import hashlib


def column_seed(global_seed: int, table_name: str, column_name: str) -> int:
    """Derive a deterministic seed for a specific column.

    The seed depends only on the global seed, table name, and column
    name.  Adding or removing other columns/tables does not change
    existing seeds — column-level independence is guaranteed.

    Returns a non-negative integer < 2^31 (compatible with both
    NumPy and stdlib ``random``).
    """
    key = f"{global_seed}:{table_name}:{column_name}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)
