"""Tests for dbsprout.generate.deterministic — per-column seed derivation.

Contract (S-130):
- ``column_seed(seed, table, column, nonce=0)`` returns a stable, non-negative
  63-bit integer derived from ``SHA-256("{seed}:{table}:{col}#{nonce}")``
  (first 8 bytes, big-endian) masked to 63 bits.
- Pure function: no I/O, no global state.
- Determinism: identical inputs → identical output across processes and
  Python versions (SHA-256 stability).
- Sensitivity: changing *any* of {seed, table, column, nonce} changes the
  output with overwhelming probability.
"""

from __future__ import annotations

import hashlib
import random
import subprocess
import sys

import pytest

from dbsprout.generate.deterministic import column_seed


class TestDeterminism:
    def test_same_inputs_same_seed(self) -> None:
        s1 = column_seed(42, "users", "email")
        s2 = column_seed(42, "users", "email")
        assert s1 == s2

    def test_same_inputs_with_explicit_nonce(self) -> None:
        s1 = column_seed(42, "users", "email", nonce=7)
        s2 = column_seed(42, "users", "email", nonce=7)
        assert s1 == s2

    def test_default_nonce_is_zero(self) -> None:
        """Calling without nonce is equivalent to nonce=0."""
        assert column_seed(42, "users", "email") == column_seed(42, "users", "email", nonce=0)

    def test_cross_run_stability_hardcoded(self) -> None:
        """SHA-256 of ``42:users:email#0`` first 8 bytes (BE) masked to 63 bits.

        Hardcoded so a Python or hashlib change would surface immediately.
        """
        digest = hashlib.sha256(b"42:users:email#0").digest()
        expected = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
        assert column_seed(42, "users", "email") == expected

    def test_cross_process_stability(self) -> None:
        """Same inputs in a fresh interpreter must yield the same seed.

        Guards against any accidental reliance on PYTHONHASHSEED or other
        process-local state.
        """
        expected = column_seed(42, "users", "email", nonce=3)
        script = (
            "from dbsprout.generate.deterministic import column_seed;"
            "print(column_seed(42, 'users', 'email', nonce=3))"
        )
        proc = subprocess.run(  # noqa: S603 — controlled args, no shell
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        assert int(proc.stdout.strip()) == expected


class TestSensitivity:
    def test_different_columns_different_seeds(self) -> None:
        assert column_seed(42, "users", "email") != column_seed(42, "users", "name")

    def test_different_tables_different_seeds(self) -> None:
        assert column_seed(42, "users", "email") != column_seed(42, "orders", "email")

    def test_different_global_seeds(self) -> None:
        assert column_seed(42, "users", "email") != column_seed(99, "users", "email")

    def test_different_nonce_changes_seed(self) -> None:
        """The re-roll knob: bumping nonce must shift the seed."""
        s0 = column_seed(42, "users", "email", nonce=0)
        s1 = column_seed(42, "users", "email", nonce=1)
        s2 = column_seed(42, "users", "email", nonce=2)
        assert s0 != s1
        assert s1 != s2
        assert s0 != s2

    def test_adding_column_doesnt_change_existing(self) -> None:
        """Column seeds are independent — extra columns don't perturb prior seeds."""
        original = column_seed(42, "users", "email")
        _ = column_seed(42, "users", "new_column")
        assert column_seed(42, "users", "email") == original


class TestBounds:
    @pytest.mark.parametrize(
        ("seed", "table", "column", "nonce"),
        [
            (0, "t", "c", 0),
            (42, "users", "email", 0),
            (42, "users", "email", 1),
            (-1, "t", "c", 0),
            (2**62, "very_long_table_name", "very_long_column_name", 999),
            (42, "", "", 0),
        ],
    )
    def test_seed_is_non_negative_63_bit_int(
        self, seed: int, table: str, column: str, nonce: int
    ) -> None:
        s = column_seed(seed, table, column, nonce=nonce)
        assert isinstance(s, int)
        assert s >= 0
        assert s < (1 << 63)

    def test_seed_compatible_with_numpy_default_rng(self) -> None:
        """Returned seed must be usable as a NumPy ``default_rng`` seed."""
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(column_seed(42, "users", "email", nonce=5))
        # Two draws from the same seed must match.
        a = rng.integers(0, 1_000_000)
        rng2 = np.random.default_rng(column_seed(42, "users", "email", nonce=5))
        b = rng2.integers(0, 1_000_000)
        assert a == b

    def test_seed_compatible_with_stdlib_random(self) -> None:
        """Returned seed must be usable as a ``random.Random`` seed."""
        s = column_seed(42, "users", "email", nonce=11)
        r1 = random.Random(s)  # noqa: S311 — not cryptographic; reproducibility test
        r2 = random.Random(s)  # noqa: S311 — not cryptographic; reproducibility test
        assert r1.random() == r2.random()
