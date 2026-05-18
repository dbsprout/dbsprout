"""Fidelity hardening: Frobenius bound + missing-scipy consistency (S-094)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import dbsprout.quality.fidelity as fid


def test_correlation_similarity_uses_tight_frobenius_bound() -> None:
    pytest.importorskip("scipy", reason="scipy not installed ([stats] extra)")
    real = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        "c": [5.0, 4.0, 3.0, 2.0, 1.0],
    }
    syn = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [1.0, 2.0, 4.0, 3.0, 5.0],
        "c": [5.0, 3.0, 3.0, 2.0, 1.0],
    }
    score = fid.correlation_similarity(real, syn)

    # Recompute with the correct tight bound 2*sqrt(n*(n-1)); the score must
    # match it, not the old loose math.sqrt(2*n*n) bound.
    names = sorted(set(real) & set(syn))
    n = len(names)
    rm = np.nan_to_num(np.corrcoef([real[c] for c in names]), nan=0.0)
    sm = np.nan_to_num(np.corrcoef([syn[c] for c in names]), nan=0.0)
    frob = float(np.sqrt(np.sum((rm - sm) ** 2)))
    expected = max(0.0, 1.0 - frob / (2.0 * math.sqrt(n * (n - 1))))
    old_bound_score = max(0.0, 1.0 - frob / math.sqrt(2.0 * n * n))

    assert score == pytest.approx(expected, abs=1e-9)
    assert score != pytest.approx(old_bound_score, abs=1e-6)


def test_correlation_similarity_clamps_at_zero() -> None:
    pytest.importorskip("scipy", reason="scipy not installed ([stats] extra)")
    real = {"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]}
    syn = {"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]}
    assert fid.correlation_similarity(real, syn) == pytest.approx(0.0, abs=1e-9)


def test_correlation_similarity_identical_is_one() -> None:
    pytest.importorskip("scipy", reason="scipy not installed ([stats] extra)")
    cols = {"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 1.0, 4.0, 3.0]}
    assert fid.correlation_similarity(cols, cols) == pytest.approx(1.0, abs=1e-9)


def test_ks_complement_raises_without_scipy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fid, "ks_2samp", None)
    with pytest.raises(ImportError, match=r'pip install "dbsprout\[stats\]"'):
        fid.ks_complement([1.0, 2.0], [3.0, 4.0])
