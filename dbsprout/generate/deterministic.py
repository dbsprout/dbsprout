"""Deterministic per-column seed derivation.

Uses SHA-256 (not Python ``hash()``) so seeds are stable across runs,
Python versions, and platforms regardless of ``PYTHONHASHSEED``.

The ``nonce`` parameter is a re-roll counter: bumping it yields a fresh
seed for the same (seed, table, column) triple, which lets the Studio UI
regenerate a single column to a different result without disturbing any
of the other column seeds (S-129 consumes this).
"""

from __future__ import annotations

import hashlib

# Mask to a non-negative 63-bit integer.  Compatible with both
# ``random.Random`` and NumPy ``default_rng``.
_MASK_63 = (1 << 63) - 1


def column_seed(
    global_seed: int,
    table_name: str,
    column_name: str,
    nonce: int = 0,
) -> int:
    """Derive a deterministic seed for a specific column.

    The seed depends only on the global seed, table name, column name,
    and an optional ``nonce``.  Adding or removing other columns/tables
    does not change existing seeds — column-level independence is
    guaranteed.  Bumping ``nonce`` lets callers re-roll a single column
    without altering anything else.

    Formula::

        SHA-256("{global_seed}:{table_name}:{column_name}#{nonce}")
            → first 8 bytes (big-endian) → int → mask to 63 bits

    Returns a non-negative 63-bit integer.  Pure function — no I/O,
    no global state.
    """
    key = f"{global_seed}:{table_name}:{column_name}#{nonce}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & _MASK_63
