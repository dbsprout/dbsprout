"""Integrity validation — FK satisfaction, PK/UNIQUE uniqueness, NOT NULL.

Validates generated data against schema constraints and returns a
detailed report with per-check pass/fail results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema, TableSchema


@dataclass(frozen=True)
class CheckResult:
    """Result of a single integrity check."""

    check: str
    table: str
    column: str
    passed: bool
    details: str = ""


@dataclass(frozen=True)
class IntegrityReport:
    """Overall integrity validation report."""

    checks: list[CheckResult] = field(default_factory=list)
    passed: bool = True


def validate_integrity(
    tables_data: dict[str, list[dict[str, Any]]],
    schema: DatabaseSchema,
) -> IntegrityReport:
    """Validate integrity of generated data against schema constraints.

    Checks FK satisfaction, PK uniqueness, UNIQUE constraints, and NOT NULL.
    Returns an ``IntegrityReport`` with per-check results.
    """
    checks: list[CheckResult] = []

    for table in schema.tables:
        rows = tables_data.get(table.name, [])
        if not rows:
            continue

        checks.extend(_check_pk_uniqueness(table, rows))
        checks.extend(_check_unique(table, rows))
        checks.extend(_check_not_null(table, rows))
        checks.extend(_check_fk_satisfaction(table, tables_data))

    passed = all(c.passed for c in checks)
    return IntegrityReport(checks=checks, passed=passed)


def _check_fk_satisfaction(
    table: TableSchema,
    tables_data: dict[str, list[dict[str, Any]]],
) -> list[CheckResult]:
    """Check that every FK value references a valid parent PK."""
    results: list[CheckResult] = []
    rows = tables_data.get(table.name, [])

    for fk in table.foreign_keys:
        parent_rows = tables_data.get(fk.ref_table, [])
        if not parent_rows:
            # Parent not generated (deferred/excluded) — skip
            continue

        # Build parent PK set
        parent_pks: set[Any] = set()
        for pr in parent_rows:
            if len(fk.ref_columns) == 1:
                parent_pks.add(pr[fk.ref_columns[0]])
            else:
                parent_pks.add(tuple(pr[c] for c in fk.ref_columns))

        # Check each FK value
        violations = 0
        for row in rows:
            if len(fk.columns) == 1:
                val = row.get(fk.columns[0])
                if val is None:
                    continue  # NULL FK is OK
                if val not in parent_pks:
                    violations += 1
            else:
                vals = tuple(row.get(c) for c in fk.columns)
                if any(v is None for v in vals):
                    continue
                if vals not in parent_pks:
                    violations += 1

        col_str = ", ".join(fk.columns)
        passed = violations == 0
        details = "" if passed else f"{violations} orphaned FK values"
        results.append(
            CheckResult(
                check="fk_satisfaction",
                table=table.name,
                column=col_str,
                passed=passed,
                details=details,
            )
        )

    return results


def _check_pk_uniqueness(
    table: TableSchema,
    rows: list[dict[str, Any]],
) -> list[CheckResult]:
    """Check that primary key values are unique."""
    if not table.primary_key:
        return []

    pk_cols = table.primary_key
    if len(pk_cols) == 1:
        values = [row[pk_cols[0]] for row in rows]
    else:
        values = [tuple(row[c] for c in pk_cols) for row in rows]

    unique_count = len(set(values))
    duplicates = len(values) - unique_count
    passed = duplicates == 0
    col_str = ", ".join(pk_cols)
    details = "" if passed else f"{duplicates} duplicate PK values"

    return [
        CheckResult(
            check="pk_uniqueness",
            table=table.name,
            column=col_str,
            passed=passed,
            details=details,
        )
    ]


def _check_unique(
    table: TableSchema,
    rows: list[dict[str, Any]],
) -> list[CheckResult]:
    """Check UNIQUE constraints (single-column and composite indexes)."""
    results: list[CheckResult] = []

    # Single-column UNIQUE
    for col in table.columns:
        if not col.unique:
            continue
        # Exclude None (SQL NULL semantics: NULL != NULL)
        values = [row[col.name] for row in rows if row.get(col.name) is not None]
        duplicates = len(values) - len(set(values))
        passed = duplicates == 0
        details = "" if passed else f"{duplicates} duplicate values"
        results.append(
            CheckResult(
                check="unique",
                table=table.name,
                column=col.name,
                passed=passed,
                details=details,
            )
        )

    # Composite UNIQUE indexes
    for idx in table.indexes:
        if not idx.unique or len(idx.columns) <= 1:
            continue
        values = [
            tuple(row[c] for c in idx.columns)
            for row in rows
            if all(row.get(c) is not None for c in idx.columns)
        ]
        duplicates = len(values) - len(set(values))
        passed = duplicates == 0
        col_str = ", ".join(idx.columns)
        details = "" if passed else f"{duplicates} duplicate tuples"
        results.append(
            CheckResult(
                check="unique",
                table=table.name,
                column=col_str,
                passed=passed,
                details=details,
            )
        )

    return results


def _check_not_null(
    table: TableSchema,
    rows: list[dict[str, Any]],
) -> list[CheckResult]:
    """Check NOT NULL constraints on non-nullable columns."""
    results: list[CheckResult] = []
    fk_cols = {col for fk in table.foreign_keys for col in fk.columns}

    for col in table.columns:
        if col.nullable:
            continue
        if col.name in fk_cols:
            continue  # FK columns checked by FK satisfaction
        null_count = sum(1 for row in rows if row.get(col.name) is None)
        passed = null_count == 0
        details = "" if passed else f"{null_count} NULL values"
        results.append(
            CheckResult(
                check="not_null",
                table=table.name,
                column=col.name,
                passed=passed,
                details=details,
            )
        )

    return results
