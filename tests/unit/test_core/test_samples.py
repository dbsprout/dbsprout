"""Tests for the bundled sample-schema registry (P1a)."""

from __future__ import annotations

import pytest

from dbsprout.core.samples import SampleInfo, list_samples, load_sample
from dbsprout.schema.models import DatabaseSchema


def test_list_samples_returns_bundled_demos() -> None:
    samples = list_samples()
    names = {s.name for s in samples}
    assert {"ecommerce", "saas"} <= names
    for s in samples:
        assert isinstance(s, SampleInfo)
        assert s.table_count > 0
        assert s.title
        assert s.description


def test_load_sample_returns_schema_with_tables() -> None:
    schema = load_sample("ecommerce")
    assert isinstance(schema, DatabaseSchema)
    assert len(schema.tables) > 0


def test_load_sample_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        load_sample("does-not-exist")
