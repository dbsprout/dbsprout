"""Shared data models for output writers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InsertResult:
    """Result of a direct database insertion."""

    tables_inserted: int
    total_rows: int
    duration_seconds: float
