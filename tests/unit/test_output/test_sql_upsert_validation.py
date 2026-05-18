"""build_upsert pk_columns subset validation (S-094 / S-045 review)."""

from __future__ import annotations

import pytest

from dbsprout.output.sql_writer import build_upsert, get_dialect_config


def test_build_upsert_rejects_pk_not_subset_of_columns() -> None:
    cfg = get_dialect_config("postgresql")
    with pytest.raises(ValueError, match=r"not a subset"):
        build_upsert("t", ["a", "b"], [{"a": 1, "b": 2}], cfg, ["a", "missing"])


def test_build_upsert_accepts_pk_subset_of_columns() -> None:
    cfg = get_dialect_config("postgresql")
    sql = build_upsert("t", ["a", "b"], [{"a": 1, "b": 2}], cfg, ["a"])
    assert "ON CONFLICT" in sql


def test_build_upsert_empty_pk_still_falls_back_to_insert() -> None:
    cfg = get_dialect_config("postgresql")
    sql = build_upsert("t", ["a", "b"], [{"a": 1, "b": 2}], cfg, [])
    assert sql.startswith("INSERT INTO")
    assert "ON CONFLICT" not in sql
