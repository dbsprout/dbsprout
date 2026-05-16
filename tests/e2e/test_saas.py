"""E2E: SaaS multi-tenant schema (tenant-isolation FKs, composite PKs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dbsprout.schema.models import ColumnType
from tests.e2e._pipeline import (
    assert_full_integrity,
    assert_programmatic_integrity,
    run_pipeline,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "schemas" / "saas.sql"


@pytest.mark.integration
def test_saas_full_integrity(tmp_path: Path) -> None:
    result = run_pipeline(FIXTURE, tmp_path, rows=50, seed=42)
    assert_full_integrity(result)
    assert_programmatic_integrity(result)
    assert len(result.schema.tables) >= 10


@pytest.mark.integration
def test_saas_tenant_isolation_and_enum(tmp_path: Path) -> None:
    result = run_pipeline(FIXTURE, tmp_path, rows=50, seed=42)
    schema = result.schema

    tm = schema.get_table("team_members")
    assert tm is not None
    assert len(tm.primary_key) == 2, tm.primary_key

    tasks = schema.get_table("tasks")
    assert tasks is not None
    assert any(fk.ref_table == "tenants" for fk in tasks.foreign_keys)
    status = next(c for c in tasks.columns if c.name == "status")
    assert status.data_type is ColumnType.ENUM
    assert status.enum_values
