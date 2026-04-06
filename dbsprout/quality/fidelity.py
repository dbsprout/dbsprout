"""Fidelity metrics — KS complement, TV complement, correlation similarity.

Compares synthetic data distributions against real reference data
to quantify how realistic the generated data is.
"""

from __future__ import annotations

import contextlib
import csv
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from dbsprout.schema.models import ColumnType

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.schema.models import DatabaseSchema

try:
    from scipy.stats import ks_2samp  # type: ignore[import-not-found,import-untyped,unused-ignore]
except ImportError:
    ks_2samp = None  # type: ignore[assignment,unused-ignore]

_NUMERIC_TYPES = frozenset(
    {
        ColumnType.INTEGER,
        ColumnType.BIGINT,
        ColumnType.SMALLINT,
        ColumnType.FLOAT,
        ColumnType.DECIMAL,
    }
)

_CATEGORICAL_TYPES = frozenset(
    {
        ColumnType.VARCHAR,
        ColumnType.TEXT,
        ColumnType.ENUM,
        ColumnType.BOOLEAN,
    }
)


@dataclass(frozen=True)
class FidelityMetric:
    """Result of a single fidelity metric calculation."""

    metric: str
    table: str
    column: str
    score: float
    details: str = ""


@dataclass(frozen=True)
class FidelityReport:
    """Overall fidelity validation report."""

    metrics: list[FidelityMetric] = field(default_factory=list)
    overall_score: float = 0.0
    passed: bool = True


# ── Metric functions ────────────────────────────────────────────────


def ks_complement(real: list[float], synthetic: list[float]) -> float:
    """KS complement: 1 - KS statistic. Higher = more similar distributions."""
    if not real or not synthetic:
        return 1.0
    if ks_2samp is None:
        return 0.0
    stat, _ = ks_2samp(real, synthetic)
    return 1.0 - float(stat)


def tv_complement(real: list[Any], synthetic: list[Any]) -> float:
    """TV complement: 1 - total variation distance. Higher = more similar."""
    if not real or not synthetic:
        return 1.0
    real_counts = Counter(real)
    syn_counts = Counter(synthetic)
    all_keys = set(real_counts) | set(syn_counts)

    real_total = len(real)
    syn_total = len(synthetic)

    distance = sum(
        abs(real_counts.get(k, 0) / real_total - syn_counts.get(k, 0) / syn_total) for k in all_keys
    )
    return 1.0 - 0.5 * distance


def correlation_similarity(
    real_cols: dict[str, list[float]],
    synthetic_cols: dict[str, list[float]],
) -> float:
    """Compare pairwise correlation matrices. Higher = more similar structure."""
    col_names = sorted(set(real_cols) & set(synthetic_cols))
    if len(col_names) < 2:
        return 1.0

    real_matrix = np.corrcoef([real_cols[c] for c in col_names])
    syn_matrix = np.corrcoef([synthetic_cols[c] for c in col_names])

    # Handle NaN from constant columns
    real_clean = np.nan_to_num(real_matrix, nan=0.0)
    syn_clean = np.nan_to_num(syn_matrix, nan=0.0)

    diff = real_clean - syn_clean
    n = len(col_names)
    max_norm = 2.0 * n * n
    frobenius = float(np.sqrt(np.sum(diff**2)))

    return max(0.0, 1.0 - frobenius / math.sqrt(max_norm))


def cardinality_similarity(real: list[Any], synthetic: list[Any]) -> float:
    """Compare unique value count ratios. Higher = more similar cardinality."""
    if not real and not synthetic:
        return 1.0
    if not real or not synthetic:
        return 1.0

    real_unique = len(set(real))
    syn_unique = len(set(synthetic))

    if real_unique == 0 and syn_unique == 0:
        return 1.0

    max_unique = max(real_unique, syn_unique)
    return 1.0 - abs(real_unique - syn_unique) / max_unique


# ── Reference data loading ──────────────────────────────────────────


def _coerce_value(value: str) -> int | float | str:
    """Try to coerce a CSV string to int, float, or leave as string."""
    if not value:
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_reference_csv(path: Path, table_name: str) -> list[dict[str, Any]]:  # noqa: ARG001
    """Load reference data from a CSV file with automatic type coercion."""
    if not path.exists():
        msg = f"Reference data file not found: {path}"
        raise FileNotFoundError(msg)

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: _coerce_value(v) for k, v in row.items()})
    return rows


# ── Orchestrator ────────────────────────────────────────────────────


def _extract_column_values(rows: list[dict[str, Any]], column: str) -> list[Any]:
    """Extract non-None values for a column from row data."""
    return [row[column] for row in rows if column in row and row[column] is not None]


def _extract_numeric_values(rows: list[dict[str, Any]], column: str) -> list[float]:
    """Extract numeric values for a column, skipping non-numeric."""
    values: list[float] = []
    for row in rows:
        v = row.get(column)
        if v is not None:
            with contextlib.suppress(ValueError, TypeError):
                values.append(float(v))
    return values


def validate_fidelity(
    synthetic_data: dict[str, list[dict[str, Any]]],
    reference_data: dict[str, list[dict[str, Any]]],
    schema: DatabaseSchema,
    threshold: float = 0.7,
) -> FidelityReport:
    """Validate fidelity of synthetic data against reference distributions."""
    if ks_2samp is None:
        msg = "scipy is required for fidelity metrics. Install it with: pip install dbsprout[stats]"
        raise ImportError(msg)

    metrics: list[FidelityMetric] = []

    for table in schema.tables:
        syn_rows = synthetic_data.get(table.name, [])
        ref_rows = reference_data.get(table.name, [])
        if not syn_rows or not ref_rows:
            continue

        numeric_cols: dict[str, ColumnType] = {}

        for col in table.columns:
            syn_vals = _extract_column_values(syn_rows, col.name)
            ref_vals = _extract_column_values(ref_rows, col.name)
            if not syn_vals or not ref_vals:
                continue

            if col.data_type in _NUMERIC_TYPES:
                numeric_cols[col.name] = col.data_type
                syn_nums = _extract_numeric_values(syn_rows, col.name)
                ref_nums = _extract_numeric_values(ref_rows, col.name)
                if syn_nums and ref_nums:
                    score = ks_complement(ref_nums, syn_nums)
                    metrics.append(
                        FidelityMetric(
                            metric="ks_complement",
                            table=table.name,
                            column=col.name,
                            score=score,
                            details=f"ks_score={score:.3f}",
                        )
                    )

            elif col.data_type in _CATEGORICAL_TYPES:
                score = tv_complement(ref_vals, syn_vals)
                metrics.append(
                    FidelityMetric(
                        metric="tv_complement",
                        table=table.name,
                        column=col.name,
                        score=score,
                        details=f"tv_score={score:.3f}",
                    )
                )

            # Cardinality for all column types
            card_score = cardinality_similarity(ref_vals, syn_vals)
            metrics.append(
                FidelityMetric(
                    metric="cardinality_similarity",
                    table=table.name,
                    column=col.name,
                    score=card_score,
                    details=f"card_score={card_score:.3f}",
                )
            )

        # Correlation similarity across numeric columns
        if len(numeric_cols) >= 2:
            ref_numeric = {c: _extract_numeric_values(ref_rows, c) for c in numeric_cols}
            syn_numeric = {c: _extract_numeric_values(syn_rows, c) for c in numeric_cols}
            corr_score = correlation_similarity(ref_numeric, syn_numeric)
            col_list = ", ".join(sorted(numeric_cols))
            metrics.append(
                FidelityMetric(
                    metric="correlation_similarity",
                    table=table.name,
                    column=col_list,
                    score=corr_score,
                    details=f"corr_score={corr_score:.3f}",
                )
            )

    overall = sum(m.score for m in metrics) / len(metrics) if metrics else 0.0
    passed = overall >= threshold
    return FidelityReport(metrics=metrics, overall_score=overall, passed=passed)
