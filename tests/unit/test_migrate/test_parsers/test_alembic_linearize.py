"""Revision-DAG linearization tests for the Alembic migration parser.

Covers:
- TestLinearize — linearizing a revision DAG into a deterministic ordered list,
  including merge/branch + cycle handling.
- TestWalker — upgrade-body walker semantics against synthetic op statements.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from .conftest import run_upgrade_body


class TestLinearize:
    def _rev(self, rid: str, down: str | None) -> object:
        from dbsprout.migrate.parsers.alembic import _Revision  # noqa: PLC0415

        return _Revision(
            path=Path(f"{rid}.py"),
            revision=rid,
            down_revision=down,
            module=ast.parse(""),
        )

    def test_empty(self) -> None:
        from dbsprout.migrate.parsers.alembic import _linearize_revisions  # noqa: PLC0415

        assert _linearize_revisions([]) == []

    def test_linear_chain(self) -> None:
        from dbsprout.migrate.parsers.alembic import _linearize_revisions  # noqa: PLC0415

        revs = [self._rev("c", "b"), self._rev("a", None), self._rev("b", "a")]
        ordered = _linearize_revisions(revs)
        assert [r.revision for r in ordered] == ["a", "b", "c"]

    def test_multiple_heads(self) -> None:
        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _linearize_revisions  # noqa: PLC0415

        revs = [self._rev("a", None), self._rev("b", "a"), self._rev("c", "a")]
        with pytest.raises(MigrationParseError, match="head"):
            _linearize_revisions(revs)

    def test_multiple_roots(self) -> None:
        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _linearize_revisions  # noqa: PLC0415

        revs = [self._rev("a", None), self._rev("b", None)]
        with pytest.raises(MigrationParseError, match=r"head|root"):
            _linearize_revisions(revs)


class TestWalker:
    def test_unknown_verb_is_skipped(self) -> None:
        # op.execute is intentionally unmapped
        changes = run_upgrade_body('    op.execute("SELECT 1")')
        assert changes == []

    def test_non_op_call_ignored(self) -> None:
        changes = run_upgrade_body('    print("hi")')
        assert changes == []

    def test_missing_upgrade_raises(self) -> None:
        from dbsprout.migrate.parsers import MigrationParseError  # noqa: PLC0415
        from dbsprout.migrate.parsers.alembic import _parse_upgrade, _Revision  # noqa: PLC0415

        module = ast.parse('revision = "r"\ndown_revision = None\n')
        rev = _Revision(path=Path("r.py"), revision="r", down_revision=None, module=module)
        with pytest.raises(MigrationParseError, match="upgrade"):
            _parse_upgrade(rev)
