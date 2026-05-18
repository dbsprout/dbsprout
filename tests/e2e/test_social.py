"""E2E: social media schema (self-ref follows, M2M, nullable FKs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e._pipeline import (
    assert_full_integrity,
    assert_programmatic_integrity,
    run_pipeline,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "schemas" / "social.sql"


@pytest.mark.integration
def test_social_full_integrity(tmp_path: Path) -> None:
    result = run_pipeline(FIXTURE, tmp_path, rows=60, seed=42)
    assert_full_integrity(result)
    assert_programmatic_integrity(result)
    assert len(result.schema.tables) >= 10


@pytest.mark.integration
def test_social_self_ref_and_nullable_fk(tmp_path: Path) -> None:
    result = run_pipeline(FIXTURE, tmp_path, rows=60, seed=42)
    schema = result.schema

    follows = schema.get_table("follows")
    assert follows is not None
    assert len(follows.primary_key) == 2
    assert all(fk.ref_table == "users" for fk in follows.foreign_keys)

    comments = schema.get_table("comments")
    assert comments is not None
    parent_fk = next(c for c in comments.columns if c.name == "parent_comment_id")
    assert parent_fk.nullable is True
    self_ref = [fk for fk in comments.foreign_keys if fk.ref_table == "comments"]
    assert self_ref, "comments must self-reference for threaded replies"
    null_seen = any(row["parent_comment_id"] is None for row in result.seed_data["comments"])
    assert null_seen, "expected some NULL nullable-FK values"
