"""Shared data models for output writers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InsertResult:
    """Result of a direct database insertion."""

    tables_inserted: int
    total_rows: int
    duration_seconds: float


@dataclass(frozen=True)
class ColumnUpdateResult:
    """Result of a column-update writer call (S-138).

    Attributes
    ----------
    rows_updated:
        Number of rows whose target column value was updated.
    duration_seconds:
        Wall-clock time the writer spent inside the transaction.
    """

    rows_updated: int
    duration_seconds: float
