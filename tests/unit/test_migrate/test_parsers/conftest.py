"""Shared fixtures and helpers for migration-parser tests.

Factored out of the S-056 monolith so future Django/Flyway/Liquibase/Prisma
parser tests can reuse the same scaffolding.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision

if TYPE_CHECKING:
    from dbsprout.migrate.models import SchemaChange

FIXTURE_PROJECT_PATH = Path(__file__).parent / "fixtures" / "alembic_project"


def run_upgrade_body(body: str) -> list[SchemaChange]:
    """Parse a synthetic Alembic upgrade body, returning the SchemaChange list.

    Replaces the `_run(body)` / `_parse_upgrade_body(body)` helpers that were
    duplicated across 6+ test classes in the pre-split monolith.
    """
    src = f'revision = "r"\ndown_revision = None\n\ndef upgrade():\n{body}\n'
    module = ast.parse(src)
    rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
    return _parse_upgrade(rev)
