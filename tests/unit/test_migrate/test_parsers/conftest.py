"""Shared fixtures and helpers for migration-parser tests.

Factored out of the S-056 monolith so future Django/Flyway/Liquibase/Prisma
parser tests can reuse the same scaffolding.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbsprout.migrate.models import SchemaChange, SchemaChangeType

FIXTURE_PROJECT_PATH = Path(__file__).parent / "fixtures" / "alembic_project"


# ---------------------------------------------------------------------------
# Django helpers
# ---------------------------------------------------------------------------

EMPTY_MIG = (
    "from django.db import migrations\n\n"
    "class Migration(migrations.Migration):\n"
    "    dependencies = []\n"
    "    operations = []\n"
)


def build_django_project(
    tmp_path: Path,
    apps: dict[str, list[tuple[str, str]]],
) -> Path:
    """Write a minimal Django project tree to ``tmp_path``.

    ``apps`` maps ``app_label`` to a list of ``(migration_stem, file_body)``
    pairs. An empty ``__init__.py`` is written next to every migrations set.
    Returns the project root (equal to ``tmp_path``).
    """
    for app_label, migrations in apps.items():
        mig_dir = tmp_path / app_label / "migrations"
        mig_dir.mkdir(parents=True, exist_ok=True)
        (mig_dir / "__init__.py").write_text("", encoding="utf-8")
        for stem, body in migrations:
            (mig_dir / f"{stem}.py").write_text(body, encoding="utf-8")
    return tmp_path


def assert_change(
    change: SchemaChange,
    *,
    change_type: SchemaChangeType,
    table_name: str,
    column_name: str | None = None,
) -> None:
    """Assert the primary identity of a ``SchemaChange``.

    Deliberately narrow — ``detail`` comparisons happen inline in tests so the
    assertion failure points at the exact key that diverged.
    """
    assert change.change_type is change_type, change
    assert change.table_name == table_name, change
    assert change.column_name == column_name, change


# ---------------------------------------------------------------------------
# Alembic helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Flyway helpers
# ---------------------------------------------------------------------------


def build_flyway_project(
    tmp_path: Path,
    migrations: dict[str, str],
    location: str = "db/migration",
) -> Path:
    """Write a minimal Flyway project tree to ``tmp_path``.

    ``migrations`` maps ``file_stem`` (e.g. ``"V1__initial"``) to the SQL body.
    ``location`` is the path (relative to ``tmp_path``) where files are placed.
    Returns the project root (equal to ``tmp_path``).
    """
    mig_dir = tmp_path / location
    mig_dir.mkdir(parents=True, exist_ok=True)
    for stem, body in migrations.items():
        (mig_dir / f"{stem}.sql").write_text(body, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Alembic helpers
# ---------------------------------------------------------------------------


def run_upgrade_body(body: str) -> list[SchemaChange]:
    """Parse a synthetic Alembic upgrade body, returning the SchemaChange list.

    Replaces the `_run(body)` / `_parse_upgrade_body(body)` helpers that were
    duplicated across 6+ test classes in the pre-split monolith.

    Imports from ``dbsprout.migrate.parsers.alembic`` are deferred into the
    function body on purpose: eager import at module scope pulls in
    ``dbsprout.migrate.__init__`` (and therefore ``numpy``) at collection time,
    which under ``--cov`` tracing causes a ``cannot load module more than once``
    crash. Lazy import mirrors the ``# noqa: PLC0415`` pattern used by the
    pre-split monolith.
    """
    from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

    src = f'revision = "r"\ndown_revision = None\n\ndef upgrade():\n{body}\n'
    module = ast.parse(src)
    rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
    return _parse_upgrade(rev)
