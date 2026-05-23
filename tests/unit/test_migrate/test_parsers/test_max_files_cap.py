"""Parametrised tests for the total-file-count cap in migration parsers.

S-105 — adds ``max_files`` to AlembicParser, DjangoMigrationParser, and
FlywayMigrationParser.  When discovered files exceed the cap, each parser
must raise ``MigrationParseError`` with a message that guides the user to
either raise the cap or narrow the scan path.

The fixture creates ``cap + 1`` tiny migration files so the test verifies the
cap fires before any unbounded allocation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dbsprout.migrate.parsers import MigrationParseError

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Small cap used in all parametrised cases; 3 files allowed, 4 trigger the cap.
# ---------------------------------------------------------------------------

_CAP = 3
_OVER = _CAP + 1  # number of files written to the fixture


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _build_alembic_fixture(tmp_path: Path) -> tuple[object, Path]:
    """Return an AlembicParser(max_files=_CAP) and a project path with _OVER revisions."""
    from pathlib import Path as _Path  # noqa: PLC0415

    from dbsprout.migrate.parsers.alembic import AlembicParser  # noqa: PLC0415

    versions = _Path(tmp_path) / "alembic" / "versions"
    versions.mkdir(parents=True)

    prev: str | None = None
    for i in range(_OVER):
        rev_id = f"rev{i:04d}"
        down = f'"{prev}"' if prev is not None else "None"
        body = f'revision = "{rev_id}"\ndown_revision = {down}\n\ndef upgrade(): pass\n'
        (versions / f"{i:04d}_mig.py").write_text(body, encoding="utf-8")
        prev = rev_id

    return AlembicParser(max_files=_CAP), _Path(tmp_path)


def _build_django_fixture(tmp_path: Path) -> tuple[object, Path]:
    """Return a DjangoMigrationParser(max_files=_CAP) and a project with _OVER migrations."""
    from pathlib import Path as _Path  # noqa: PLC0415

    from dbsprout.migrate.parsers.django import DjangoMigrationParser  # noqa: PLC0415

    mig_dir = _Path(tmp_path) / "myapp" / "migrations"
    mig_dir.mkdir(parents=True)
    (mig_dir / "__init__.py").write_text("", encoding="utf-8")

    for i in range(_OVER):
        body = (
            "from django.db import migrations\n\n"
            "class Migration(migrations.Migration):\n"
            "    dependencies = []\n"
            "    operations = []\n"
        )
        (mig_dir / f"{i + 1:04d}_mig.py").write_text(body, encoding="utf-8")

    return DjangoMigrationParser(max_files=_CAP), _Path(tmp_path)


def _build_flyway_fixture(tmp_path: Path) -> tuple[object, Path]:
    """Return a FlywayMigrationParser(max_files=_CAP) and a project with _OVER migrations."""
    from pathlib import Path as _Path  # noqa: PLC0415

    from dbsprout.migrate.parsers.flyway import FlywayMigrationParser  # noqa: PLC0415

    mig_dir = _Path(tmp_path) / "db" / "migration"
    mig_dir.mkdir(parents=True)

    for i in range(_OVER):
        name = f"V{i + 1}__mig_{i}.sql"
        (mig_dir / name).write_text("-- tiny\n", encoding="utf-8")

    return FlywayMigrationParser(max_files=_CAP), _Path(tmp_path)


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    [
        _build_alembic_fixture,
        _build_django_fixture,
        _build_flyway_fixture,
    ],
    ids=["alembic", "django", "flyway"],
)
def test_max_files_cap_raises(
    tmp_path: Path,
    builder: object,
) -> None:
    """When discovered files exceed max_files, MigrationParseError is raised.

    The error message must guide the user to raise the cap or narrow the path.
    """
    assert callable(builder)
    parser, project_path = builder(tmp_path)  # type: ignore[operator]

    with pytest.raises(MigrationParseError) as exc_info:
        parser.detect_changes(project_path)  # type: ignore[union-attr]

    msg = str(exc_info.value).lower()

    # Message must contain a numeric cap reference and guidance.
    assert str(_CAP) in str(exc_info.value), (
        f"Expected cap value {_CAP} in error message, got: {exc_info.value}"
    )
    assert any(word in msg for word in ("max_files", "cap", "limit")), (
        f"Expected cap-guidance keyword in message, got: {exc_info.value}"
    )
    assert any(word in msg for word in ("raise", "narrow", "reduce", "lower")), (
        f"Expected narrowing-guidance in message, got: {exc_info.value}"
    )


def _remove_last_file(project_path: Path) -> None:
    """Remove the last file in the migration discovery area to reduce count to _CAP."""
    from pathlib import Path as _Path  # noqa: PLC0415

    pp = _Path(project_path)
    alembic_ver = pp / "alembic" / "versions"
    django_mig = pp / "myapp" / "migrations"
    flyway_mig = pp / "db" / "migration"

    if alembic_ver.exists():
        ver_files = sorted(alembic_ver.glob("*.py"))
        ver_files[-1].unlink()
    elif django_mig.exists():
        mig_files = sorted(f for f in django_mig.glob("*.py") if f.name != "__init__.py")
        mig_files[-1].unlink()
    elif flyway_mig.exists():
        sql_files = sorted(flyway_mig.glob("*.sql"))
        sql_files[-1].unlink()


def _cap_error_raised(parser: object, project_path: Path) -> bool:
    """Return True if detect_changes raises a cap-related MigrationParseError."""
    cap_words = ("max_files", "cap", "limit")
    try:
        parser.detect_changes(project_path)  # type: ignore[union-attr]
    except MigrationParseError as exc:
        return any(word in str(exc).lower() for word in cap_words)
    return False


@pytest.mark.parametrize(
    "builder",
    [
        _build_alembic_fixture,
        _build_django_fixture,
        _build_flyway_fixture,
    ],
    ids=["alembic", "django", "flyway"],
)
def test_max_files_cap_not_triggered_at_cap(
    tmp_path: Path,
    builder: object,
) -> None:
    """Exactly cap files must NOT raise a cap error — only strictly exceeding it does."""
    assert callable(builder)

    # Build fixture with _OVER files, then remove one to land at exactly _CAP.
    parser, project_path = builder(tmp_path)  # type: ignore[operator]
    _remove_last_file(project_path)

    # At exactly _CAP files the cap must NOT fire.
    assert not _cap_error_raised(parser, project_path), (
        f"Cap error unexpectedly raised at exactly {_CAP} files"
    )
