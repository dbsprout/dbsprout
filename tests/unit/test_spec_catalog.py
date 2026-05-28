"""Unit tests for the generator catalogue (S-120).

The catalogue is the *machine-derived* source of truth for the (provider,
method) pairs the Studio method-picker surfaces. It is built from the
`PATTERNS` registry in :mod:`dbsprout.spec.patterns` and the
``_TYPE_FALLBACKS`` map in :mod:`dbsprout.spec.heuristics` — no hand-rolled
string list. These tests pin the helper API the web router relies on:

* :func:`iter_methods` — unique ``(provider, method)`` entries with the
  dtypes each method applies to and the param keys it accepts.
* :func:`applies_to` — dtype filter used by the picker UI (also enforced
  on the server side at edit time by S-119's constraints guard).
* :func:`param_keys_for` — small static schema for what params a given
  method consumes; trimmed to keys a user would actually edit.
"""

from __future__ import annotations

import pytest

from dbsprout.schema.models import ColumnType
from dbsprout.spec.catalog import applies_to, iter_methods, param_keys_for
from dbsprout.spec.heuristics import _TYPE_FALLBACKS
from dbsprout.spec.patterns import PATTERNS


def test_iter_methods_yields_unique_pairs() -> None:
    """Every ``(provider, method)`` in the catalogue is unique."""
    entries = list(iter_methods())
    keys = [(e.provider, e.method) for e in entries]
    assert len(keys) == len(set(keys)), f"duplicates in catalogue: {keys}"
    assert entries, "catalogue must not be empty"


def test_iter_methods_covers_patterns_registry() -> None:
    """Every pattern in ``PATTERNS`` must appear in the catalogue.

    Closes the "no hand-maintained list" loop: deletion of a pattern in
    :mod:`dbsprout.spec.patterns` automatically shrinks the catalogue.
    """
    catalogue_keys = {(e.provider, e.method) for e in iter_methods()}
    for pattern in PATTERNS:
        assert (pattern.provider, pattern.generator_name) in catalogue_keys, (
            f"pattern {pattern.provider}.{pattern.generator_name} missing from catalogue"
        )


def test_iter_methods_covers_type_fallbacks() -> None:
    """Every type fallback method must appear in the catalogue."""
    catalogue_keys = {(e.provider, e.method) for e in iter_methods()}
    for method, provider in _TYPE_FALLBACKS.values():
        assert (provider, method) in catalogue_keys, (
            f"type fallback {provider}.{method} missing from catalogue"
        )


@pytest.mark.parametrize(
    ("provider", "method", "dtype", "expected"),
    [
        # email is a textual method — VARCHAR yes, INTEGER no.
        ("mimesis", "email", ColumnType.VARCHAR, True),
        ("mimesis", "email", ColumnType.TEXT, True),
        ("mimesis", "email", ColumnType.INTEGER, False),
        # random_int is numeric — TEXT no, INTEGER/BIGINT yes.
        ("builtin", "random_int", ColumnType.INTEGER, True),
        ("builtin", "random_int", ColumnType.BIGINT, True),
        ("builtin", "random_int", ColumnType.TEXT, False),
        ("builtin", "random_int", ColumnType.VARCHAR, False),
        # bool only on BOOLEAN.
        ("builtin", "random_bool", ColumnType.BOOLEAN, True),
        ("builtin", "random_bool", ColumnType.VARCHAR, False),
        # uuid4 → UUID column and also VARCHAR (legacy stringified ids).
        ("builtin", "uuid4", ColumnType.UUID, True),
        ("builtin", "uuid4", ColumnType.VARCHAR, True),
        # datetime → DATETIME/TIMESTAMP, not VARCHAR.
        ("mimesis", "datetime", ColumnType.DATETIME, True),
        ("mimesis", "datetime", ColumnType.TIMESTAMP, True),
        ("mimesis", "datetime", ColumnType.VARCHAR, False),
    ],
)
def test_applies_to_dtype_filter(
    provider: str,
    method: str,
    dtype: ColumnType,
    expected: bool,
) -> None:
    """Dtype filter must reject obviously-wrong (method, dtype) pairs."""
    assert applies_to(method, provider, dtype) is expected


def test_applies_to_unknown_method_false() -> None:
    """Unknown methods are rejected for every dtype."""
    assert applies_to("not_a_real_method", "mimesis", ColumnType.VARCHAR) is False


def test_param_keys_random_int_has_min_max() -> None:
    """``random_int`` advertises ``min`` / ``max`` so the picker can render inputs."""
    keys = param_keys_for("random_int")
    assert "min" in keys
    assert "max" in keys


def test_param_keys_varchar_methods_have_max_length() -> None:
    """String-shaped methods advertise ``max_length`` (driven by ``_build_params``)."""
    assert "max_length" in param_keys_for("random_string")


def test_param_keys_unknown_method_empty() -> None:
    """Unknown methods get an empty (not raised) param-key list."""
    assert param_keys_for("not_a_real_method") == frozenset()


def test_method_entry_carries_description_and_dtypes() -> None:
    """Each entry must expose the fields the JSON router echoes back."""
    sample = next(iter(iter_methods()))
    assert isinstance(sample.provider, str)
    assert isinstance(sample.method, str)
    assert isinstance(sample.description, str)
    assert isinstance(sample.dtypes, frozenset)
    assert all(isinstance(d, ColumnType) for d in sample.dtypes)
    assert isinstance(sample.params, frozenset)
