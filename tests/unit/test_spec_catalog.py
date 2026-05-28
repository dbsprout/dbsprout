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


# ── S-123: human-readable method descriptions ─────────────────────────────


def _entry(provider: str, method: str):  # type: ignore[no-untyped-def]
    """Locate the catalogue entry for ``(provider, method)`` or fail loudly."""
    for entry in iter_methods():
        if entry.provider == provider and entry.method == method:
            return entry
    msg = f"missing catalogue entry {provider}.{method}"
    raise AssertionError(msg)


@pytest.mark.parametrize(
    ("provider", "method", "needle"),
    [
        ("mimesis", "email", "email"),
        ("builtin", "random_int", "integer"),
        ("builtin", "random_bool", "boolean"),
        ("builtin", "uuid4", "UUID"),
        ("mimesis", "datetime", "datetime"),
        ("builtin", "random_choice", "enum"),
    ],
)
def test_method_entry_description_is_user_friendly_for_known_methods(
    provider: str,
    method: str,
    needle: str,
) -> None:
    """Known methods carry a curated, user-friendly description.

    The exact wording isn't pinned; we just require the description to
    contain a topical keyword so the UI tooltip is meaningful rather
    than the legacy ``"Mimesis email"`` placeholder.
    """
    entry = _entry(provider, method)
    assert needle.lower() in entry.description.lower(), (
        f"{provider}.{method} description {entry.description!r} missing {needle!r}"
    )


def test_describe_falls_back_for_unknown_methods() -> None:
    """Unknown methods still get a non-empty default description.

    Plugin-supplied methods that aren't curated must still surface
    *something* on the UI rather than an empty string.
    """
    from dbsprout.spec.catalog import _describe  # noqa: PLC0415

    out = _describe("custom_provider", "shiny_new_method")
    assert out
    assert isinstance(out, str)
    # Fallback shape is provider-aware ("Custom_provider shiny new method")
    # — we only assert it includes both tokens.
    assert "shiny" in out.lower()


def test_every_catalogue_entry_has_non_empty_description() -> None:
    """No catalogue row should ship a blank description."""
    for entry in iter_methods():
        assert entry.description, f"{entry.provider}.{entry.method} blank description"
        assert entry.description.strip(), (
            f"{entry.provider}.{entry.method} whitespace-only description"
        )


# ── S-146: per-method example values ──────────────────────────────────────


def test_method_entry_carries_example_field() -> None:
    """Every catalogue row exposes an ``example`` string field.

    The Studio method-picker (S-120) surfaces a single illustrative value
    next to each method button so the user can recognise the generator's
    output at a glance — closing the "what does this method actually
    produce?" gap S-146 fills.
    """
    sample = next(iter(iter_methods()))
    assert hasattr(sample, "example"), "MethodEntry.example missing"
    assert isinstance(sample.example, str), type(sample.example)


def test_every_catalogue_entry_has_non_empty_example() -> None:
    """No row should ship a blank example — picker tooltip relies on it."""
    for entry in iter_methods():
        assert entry.example, f"{entry.provider}.{entry.method} blank example"
        assert entry.example.strip(), f"{entry.provider}.{entry.method} whitespace-only example"


@pytest.mark.parametrize(
    ("provider", "method", "needle"),
    [
        # Curated examples — high-traffic methods get hand-rolled values
        # so the picker UI shows something readable rather than a verbatim
        # docstring sentence.
        ("mimesis", "email", "@"),
        ("builtin", "random_int", "4"),
        ("builtin", "uuid4", "-"),
        ("mimesis", "first_name", ""),  # any non-empty curated name
    ],
)
def test_curated_examples_present_for_high_traffic_methods(
    provider: str,
    method: str,
    needle: str,
) -> None:
    """High-traffic methods carry curated example values."""
    entry = _entry(provider, method)
    if needle:
        assert needle in entry.example, (
            f"{provider}.{method} example {entry.example!r} missing {needle!r}"
        )
    else:
        assert entry.example.strip(), f"{provider}.{method} blank example"


def test_uncurated_methods_fall_back_to_description() -> None:
    """Methods without a curated example fall back to the description.

    The fallback path keeps plugin-supplied methods (or any catalogue row
    that hasn't been hand-curated) from shipping a blank example. The
    fallback is the curated description so the picker always has *some*
    user-facing copy under the method button.
    """
    from dbsprout.spec.catalog import _example_for  # noqa: PLC0415

    # Force an unknown method through the helper directly — the public
    # ``iter_methods`` doesn't expose plugin methods, but the helper is
    # the single fallback edge any new entry would hit.
    out = _example_for("plugin", "shiny_new_method")
    assert out
    assert isinstance(out, str)


def test_example_field_is_immutable_on_method_entry() -> None:
    """``MethodEntry`` stays frozen — the ``example`` field is read-only."""
    sample = next(iter(iter_methods()))
    with pytest.raises((AttributeError, Exception)):
        sample.example = "mutated"  # type: ignore[misc]
