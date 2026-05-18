"""E2E: CMS schema (self-ref pages, revisions, RBAC, enums)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dbsprout.schema.models import ColumnType
from tests.e2e._pipeline import (
    assert_full_integrity,
    assert_programmatic_integrity,
    run_pipeline,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "schemas" / "cms.sql"


@pytest.mark.integration
def test_cms_full_integrity(tmp_path: Path) -> None:
    result = run_pipeline(FIXTURE, tmp_path, rows=50, seed=42)
    assert_full_integrity(result)
    assert_programmatic_integrity(result)
    assert len(result.schema.tables) >= 10


@pytest.mark.integration
def test_cms_self_ref_rbac_and_enum(tmp_path: Path) -> None:
    result = run_pipeline(FIXTURE, tmp_path, rows=50, seed=42)
    schema = result.schema

    pages = schema.get_table("pages")
    assert pages is not None
    assert any(fk.ref_table == "pages" for fk in pages.foreign_keys)
    status = next(c for c in pages.columns if c.name == "status")
    assert status.data_type is ColumnType.ENUM
    assert status.enum_values

    rp = schema.get_table("role_permissions")
    assert rp is not None
    assert len(rp.primary_key) == 2
