"""Statistical generation engine — Gaussian copula (S-071).

Learns per-column empirical marginals plus a Gaussian copula over the
numeric/datetime block from a reference sample, then draws synthetic rows
that reproduce the source marginals and pairwise correlations.

Implemented from scratch with ``scipy`` + ``numpy`` (the dependency set
ships neither SDV nor sdmetrics). scipy/numpy are imported lazily so the
CLI startup budget is unaffected when another engine is selected.

FK columns and autoincrement primary keys are emitted as ``None`` — the
orchestrator samples FK values from parent PKs afterward, preserving 100%
referential integrity. When a table has no usable reference sample (fewer
than :data:`MIN_SAMPLE` rows) the engine logs a warning and delegates to
:class:`~dbsprout.generate.engines.heuristic.HeuristicEngine`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dbsprout.generate.deterministic import column_seed

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from dbsprout.schema.models import ColumnSchema, TableSchema
    from dbsprout.spec.models import GeneratorMapping

logger = logging.getLogger(__name__)

#: Minimum reference rows for a stable copula fit; below this the engine
#: falls back to the heuristic engine with a warning.
MIN_SAMPLE = 100

_NUMERIC_TYPES = frozenset({"integer", "bigint", "smallint", "float", "decimal"})
_DATETIME_TYPES = frozenset({"date", "datetime", "timestamp", "time"})
_INTEGER_TYPES = frozenset({"integer", "bigint", "smallint"})


# ── Marginal helpers ─────────────────────────────────────────────────


class _EmpiricalMarginal:
    """Sorted reference values supporting quantile inverse-transform."""

    __slots__ = ("sorted_values",)

    def __init__(self, sorted_values: Any) -> None:
        self.sorted_values = sorted_values


def _empirical_marginal(sample: list[float]) -> _EmpiricalMarginal:
    """Build an empirical marginal from numeric *sample* values."""
    import numpy as np  # noqa: PLC0415

    arr = np.sort(np.asarray(sample, dtype=float))
    return _EmpiricalMarginal(arr)


def _inverse_marginal(marginal: _EmpiricalMarginal, u: float) -> float:
    """Map a uniform *u* in [0, 1] back to the empirical distribution."""
    import numpy as np  # noqa: PLC0415

    values = marginal.sorted_values
    n = values.shape[0]
    if n == 0:  # pragma: no cover - guarded by callers
        return 0.0
    if n == 1:
        return float(values[0])
    pos = float(np.clip(u, 0.0, 1.0)) * (n - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _to_normal_scores(values: Any) -> Any:
    """Rank-transform *values* to approximate standard-normal scores."""
    import numpy as np  # noqa: PLC0415
    from scipy import stats  # noqa: PLC0415

    arr = np.asarray(values, dtype=float)
    n = arr.shape[0]
    ranks = stats.rankdata(arr, method="average")
    uniforms = ranks / (n + 1.0)
    return stats.norm.ppf(uniforms)


def _sample_categorical(sample: list[Any], num_rows: int, rng: np.random.Generator) -> list[Any]:
    """Draw *num_rows* categories matching *sample* frequencies."""
    import numpy as np  # noqa: PLC0415

    categories: list[Any] = []
    counts: list[int] = []
    index: dict[Any, int] = {}
    for value in sample:
        key = value
        slot = index.get(key)
        if slot is None:
            index[key] = len(categories)
            categories.append(value)
            counts.append(1)
        else:
            counts[slot] += 1
    total = float(sum(counts))
    probs = np.asarray(counts, dtype=float) / total
    picks = rng.choice(len(categories), size=num_rows, p=probs)
    return [categories[i] for i in picks]


# ── Copula helpers ───────────────────────────────────────────────────


def _nearest_psd(matrix: Any) -> Any:
    """Return the nearest positive-semidefinite correlation matrix."""
    import numpy as np  # noqa: PLC0415

    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, 1e-8, None)
    repaired = eigvecs @ np.diag(eigvals) @ eigvecs.T
    diag = np.sqrt(np.clip(np.diag(repaired), 1e-12, None))
    repaired = repaired / np.outer(diag, diag)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def _fit_copula(normal_scores: Any) -> Any:
    """Estimate the Gaussian-copula correlation matrix (PSD-repaired)."""
    import numpy as np  # noqa: PLC0415

    if normal_scores.shape[1] == 1:
        return np.array([[1.0]])
    # Zero-variance columns (e.g. a constant datetime) make ``np.corrcoef``
    # divide by zero; suppress the warning and treat them as uncorrelated.
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(normal_scores, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return _nearest_psd(corr)


# ── Engine ───────────────────────────────────────────────────────────


def _is_numeric(col: ColumnSchema) -> bool:
    return col.data_type.value in _NUMERIC_TYPES


def _is_datetime(col: ColumnSchema) -> bool:
    return col.data_type.value in _DATETIME_TYPES


def _coerce_numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_datetime_ordinal(value: Any) -> float | None:
    """Convert a datetime-ish value to a float ordinal (epoch seconds)."""
    from datetime import date, datetime, time  # noqa: PLC0415

    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, date):
        return float(value.toordinal())
    if isinstance(value, time):
        return value.hour * 3600.0 + value.minute * 60.0 + value.second
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return float(value) if isinstance(value, (int, float)) else None


class StatisticalEngine:
    """Gaussian-copula synthetic data engine."""

    def __init__(self, *, locale: str = "en", seed: int = 42) -> None:
        self._seed = seed
        self._locale = locale

    def generate_table(
        self,
        table: TableSchema,
        reference_rows: list[dict[str, Any]],
        mappings: dict[str, GeneratorMapping],
        num_rows: int,
    ) -> list[dict[str, Any]]:
        """Generate ``num_rows`` rows mirroring *reference_rows*.

        Falls back to the heuristic engine (with a warning) when the
        reference sample is too small for a stable copula fit.
        """
        if len(reference_rows) < MIN_SAMPLE:
            logger.warning(
                "Statistical engine: insufficient reference data for "
                "table %r (%d rows < %d); falling back to heuristic engine.",
                table.name,
                len(reference_rows),
                MIN_SAMPLE,
            )
            from dbsprout.generate.engines.heuristic import (  # noqa: PLC0415
                HeuristicEngine,
            )

            return HeuristicEngine(seed=self._seed).generate_table(table, mappings, num_rows)

        return self._fit_and_sample(table, reference_rows, num_rows)

    def _fit_and_sample(
        self,
        table: TableSchema,
        reference_rows: list[dict[str, Any]],
        num_rows: int,
    ) -> list[dict[str, Any]]:
        import numpy as np  # noqa: PLC0415
        from scipy import stats  # noqa: PLC0415

        seed = column_seed(self._seed, table.name, "__copula__")
        rng = np.random.default_rng(seed)

        skip = _skip_columns(table)
        col_data: dict[str, list[Any]] = {}

        numeric_cols, numeric_marginals, numeric_scores = self._fit_numeric(
            table, reference_rows, skip
        )
        for col in table.columns:
            if col.name in skip:
                col_data[col.name] = [None] * num_rows
            elif col.name not in numeric_cols:
                col_data[col.name] = self._sample_other(col, reference_rows, num_rows, rng)

        if numeric_cols:
            corr = _fit_copula(numeric_scores)
            chol = np.linalg.cholesky(corr)
            latent = rng.standard_normal(size=(num_rows, len(numeric_cols)))
            correlated = latent @ chol.T
            uniforms = stats.norm.cdf(correlated)
            for idx, col_name in enumerate(numeric_cols):
                col = _column(table, col_name)
                marg = numeric_marginals[idx]
                vals = [_inverse_marginal(marg, float(u)) for u in uniforms[:, idx]]
                col_data[col_name] = [self._cast_numeric(col, v) for v in vals]

        return [{name: col_data[name][i] for name in col_data} for i in range(num_rows)]

    def _fit_numeric(
        self,
        table: TableSchema,
        reference_rows: list[dict[str, Any]],
        skip: set[str],
    ) -> tuple[list[str], list[_EmpiricalMarginal], Any]:
        """Build marginals + normal scores for the numeric/datetime block."""
        import numpy as np  # noqa: PLC0415

        numeric_cols: list[str] = []
        marginals: list[_EmpiricalMarginal] = []
        score_columns: list[Any] = []

        for col in table.columns:
            if col.name in skip:
                continue
            if not (_is_numeric(col) or _is_datetime(col)):
                continue
            coerce = _coerce_datetime_ordinal if _is_datetime(col) else _coerce_numeric
            values = [v for v in (coerce(r.get(col.name)) for r in reference_rows) if v is not None]
            if len(values) < MIN_SAMPLE:
                continue
            numeric_cols.append(col.name)
            marginals.append(_empirical_marginal(values))
            score_columns.append(_to_normal_scores(values))

        if not numeric_cols:
            return [], [], np.empty((0, 0))

        # Columns may have differing non-null counts; truncate to the
        # shortest so the score matrix is rectangular. This is an
        # approximation for the copula correlation estimate when columns
        # have divergent null patterns (v1 limitation — marginals are
        # still fitted on each column's full non-null sample).
        width = min(len(c) for c in score_columns)
        scores = np.column_stack([c[:width] for c in score_columns])
        return numeric_cols, marginals, scores

    def _sample_other(
        self,
        col: ColumnSchema,
        reference_rows: list[dict[str, Any]],
        num_rows: int,
        rng: np.random.Generator,
    ) -> list[Any]:
        """Sample categorical / boolean / other columns by frequency."""
        sample = [r.get(col.name) for r in reference_rows if r.get(col.name) is not None]
        if not sample:
            return [None] * num_rows
        drawn = _sample_categorical(sample, num_rows, rng)
        if col.data_type.value == "boolean":
            return [bool(v) for v in drawn]
        return drawn

    @staticmethod
    def _cast_numeric(col: ColumnSchema, value: float) -> Any:
        if _is_datetime(col):
            return _ordinal_to_value(col, value)
        if col.data_type.value in _INTEGER_TYPES:
            return round(value)
        return float(value)


# ── Module helpers ───────────────────────────────────────────────────


def _skip_columns(table: TableSchema) -> set[str]:
    """FK columns + autoincrement PKs are emitted as ``None``."""
    skip = {col for fk in table.foreign_keys for col in fk.columns}
    skip |= {col.name for col in table.columns if col.primary_key and col.autoincrement}
    return skip


def _column(table: TableSchema, name: str) -> ColumnSchema:
    for col in table.columns:
        if col.name == name:
            return col
    raise KeyError(name)  # pragma: no cover - defensive


def _ordinal_to_value(col: ColumnSchema, ordinal: float) -> Any:
    """Convert a numeric ordinal back to the column's datetime type."""
    from datetime import datetime, time, timedelta, timezone  # noqa: PLC0415

    kind = col.data_type.value
    if kind == "date":
        from datetime import date  # noqa: PLC0415

        return date.fromordinal(max(1, round(ordinal)))
    if kind == "time":
        secs = max(0, min(86399, round(ordinal)))
        base = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=secs)
        return time(base.hour, base.minute, base.second)
    return datetime.fromtimestamp(ordinal, tz=timezone.utc)


__all__ = ["MIN_SAMPLE", "StatisticalEngine"]
