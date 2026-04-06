"""Detection metrics — Classifier Two-Sample Test (C2ST).

Measures how easily a classifier can distinguish synthetic data from real data.
An accuracy of 0.5 means indistinguishable (ideal); 1.0 means trivially separable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from dbsprout.schema.models import ColumnType

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema

try:
    from sklearn.linear_model import (  # type: ignore[import-not-found,import-untyped,unused-ignore]
        LogisticRegression,
    )
    from sklearn.model_selection import (  # type: ignore[import-not-found,import-untyped,unused-ignore]
        StratifiedKFold,
        cross_val_score,
    )
    from sklearn.preprocessing import (  # type: ignore[import-not-found,import-untyped,unused-ignore]
        OrdinalEncoder,
        StandardScaler,
    )
except ImportError:
    LogisticRegression = None  # type: ignore[assignment,misc,unused-ignore]
    StratifiedKFold = None  # type: ignore[assignment,misc,unused-ignore]
    cross_val_score = None  # type: ignore[assignment,unused-ignore]
    OrdinalEncoder = None  # type: ignore[assignment,misc,unused-ignore]
    StandardScaler = None  # type: ignore[assignment,misc,unused-ignore]

_MAX_SUBSAMPLE = 10_000

_NUMERIC_TYPES = frozenset(
    {
        ColumnType.INTEGER,
        ColumnType.BIGINT,
        ColumnType.SMALLINT,
        ColumnType.FLOAT,
        ColumnType.DECIMAL,
    }
)

_MIN_ROWS_PER_CLASS = 10


# ── Data models ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DetectionMetric:
    """Result of a single detection metric calculation."""

    metric: str
    table: str
    accuracy: float
    details: str = ""


@dataclass(frozen=True)
class DetectionReport:
    """Overall detection validation report."""

    metrics: list[DetectionMetric] = field(default_factory=list)
    overall_score: float = 0.5
    passed: bool = True


# ── Core C2ST function ─────────────────────────────────────────────────


def c2st_accuracy(
    real_features: np.ndarray,
    synthetic_features: np.ndarray,
    seed: int = 42,
) -> float:
    """Classifier Two-Sample Test accuracy.

    Trains a logistic regression classifier to distinguish real from synthetic
    data using 5-fold stratified cross-validation. Returns mean accuracy.

    Args:
        real_features: 2-D array of shape (n_real, n_features).
        synthetic_features: 2-D array of shape (n_synthetic, n_features).
        seed: Random seed for reproducibility.

    Returns:
        Mean cross-validated accuracy in [0, 1].
        0.5 means indistinguishable; 1.0 means trivially separable.
        Returns 0.5 for empty inputs.
    """
    if LogisticRegression is None:
        msg = (
            "scikit-learn is required for detection metrics. "
            "Install it with: pip install dbsprout[stats]"
        )
        raise ImportError(msg)

    # Guard: empty inputs
    if (
        real_features.size == 0
        or synthetic_features.size == 0
        or real_features.ndim < 2
        or synthetic_features.ndim < 2
        or real_features.shape[1] == 0
        or synthetic_features.shape[1] == 0
    ):
        return 0.5

    rng = np.random.default_rng(seed)

    # Subsample if too large
    real = _subsample(real_features, rng)
    synthetic = _subsample(synthetic_features, rng)

    # Label: real=1, synthetic=0
    y_real = np.ones(real.shape[0], dtype=np.intp)
    y_syn = np.zeros(synthetic.shape[0], dtype=np.intp)

    x_combined = np.vstack([real, synthetic])
    y_combined = np.concatenate([y_real, y_syn])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = LogisticRegression(random_state=seed, max_iter=1000)
    scores = cross_val_score(clf, x_combined, y_combined, cv=skf, scoring="accuracy")

    return float(np.mean(scores))


def _subsample(features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Subsample rows if exceeding _MAX_SUBSAMPLE."""
    if features.shape[0] <= _MAX_SUBSAMPLE:
        return features
    indices = rng.choice(features.shape[0], size=_MAX_SUBSAMPLE, replace=False)
    return features[indices]


# ── Feature matrix builder ─────────────────────────────────────────────


def _safe_float(val: Any) -> float:
    """Convert a value to float, returning 0.0 on failure."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _build_feature_matrix_combined(
    real_rows: list[dict[str, Any]],
    synthetic_rows: list[dict[str, Any]],
    columns: list[str],
    schema_columns: dict[str, ColumnType],
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrices for real and synthetic data with shared preprocessing.

    Fits StandardScaler and OrdinalEncoder on the combined dataset so both
    halves share the same feature space. This is critical for a fair C2ST.

    Args:
        real_rows: Reference/real row data.
        synthetic_rows: Synthetic row data.
        columns: Column names to include.
        schema_columns: Mapping from column name to ColumnType.

    Returns:
        Tuple of (real_features, synthetic_features), each a 2-D numpy array.
    """
    combined_rows = [*real_rows, *synthetic_rows]
    n_real = len(real_rows)

    if not combined_rows or not columns:
        empty = np.empty((0, 0))
        return empty, empty

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in columns:
        col_type = schema_columns.get(col)
        if col_type in _NUMERIC_TYPES:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    feature_blocks: list[np.ndarray] = []

    # Numeric features — vectorized extraction
    if numeric_cols:
        num_data = np.array(
            [[_safe_float(row.get(col)) for col in numeric_cols] for row in combined_rows],
            dtype=np.float64,
        )
        scaler = StandardScaler()
        scaled = scaler.fit_transform(num_data)
        feature_blocks.append(scaled)

    # Categorical features — fit encoder on combined data
    if categorical_cols:
        cat_data = [[str(row.get(col, "")) for col in categorical_cols] for row in combined_rows]
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoded = encoder.fit_transform(cat_data)
        feature_blocks.append(encoded)

    if not feature_blocks:  # pragma: no cover — defensive; unreachable with current logic
        empty = np.empty((0, 0))
        return empty, empty

    combined = np.hstack(feature_blocks)

    # Drop constant columns (zero variance)
    variances = np.var(combined, axis=0)
    non_constant_mask = variances > 0.0
    filtered = combined[:, non_constant_mask]

    return filtered[:n_real], filtered[n_real:]


# ── Orchestrator ───────────────────────────────────────────────────────


def validate_detection(
    synthetic_data: dict[str, list[dict[str, Any]]],
    reference_data: dict[str, list[dict[str, Any]]],
    schema: DatabaseSchema,
    threshold: float = 0.6,
    seed: int = 42,
) -> DetectionReport:
    """Validate how distinguishable synthetic data is from real data.

    Runs a C2ST (Classifier Two-Sample Test) per table. Lower accuracy
    is better (0.5 = indistinguishable).

    Args:
        synthetic_data: Per-table synthetic rows.
        reference_data: Per-table real/reference rows.
        schema: Database schema for column type classification.
        threshold: Maximum acceptable C2ST accuracy. ``passed`` is True
            when the overall score is at or below this threshold.
        seed: Random seed for reproducibility.

    Returns:
        DetectionReport with per-table metrics and overall assessment.
    """
    if LogisticRegression is None:
        msg = (
            "scikit-learn is required for detection metrics. "
            "Install it with: pip install dbsprout[stats]"
        )
        raise ImportError(msg)

    metrics: list[DetectionMetric] = []

    for table in schema.tables:
        syn_rows = synthetic_data.get(table.name, [])
        ref_rows = reference_data.get(table.name, [])
        if not syn_rows or not ref_rows:
            continue

        # Find shared columns between data and schema
        syn_cols = set(syn_rows[0].keys()) if syn_rows else set()
        ref_cols = set(ref_rows[0].keys()) if ref_rows else set()
        schema_col_names = {col.name for col in table.columns}
        shared_cols = sorted(syn_cols & ref_cols & schema_col_names)

        if not shared_cols:
            continue

        # Build column type mapping
        col_type_map: dict[str, ColumnType] = {
            col.name: col.data_type for col in table.columns if col.name in shared_cols
        }

        # Skip if not enough rows for 5-fold CV (need >=10 per class)
        if len(ref_rows) < _MIN_ROWS_PER_CLASS or len(syn_rows) < _MIN_ROWS_PER_CLASS:
            continue

        # Build feature matrices — fit on combined data for fair comparison
        ref_features, syn_features = _build_feature_matrix_combined(
            ref_rows, syn_rows, shared_cols, col_type_map
        )

        # Skip if no usable features
        if ref_features.shape[1] == 0 or syn_features.shape[1] == 0:
            continue

        accuracy = c2st_accuracy(ref_features, syn_features, seed=seed)
        metrics.append(
            DetectionMetric(
                metric="c2st",
                table=table.name,
                accuracy=accuracy,
                details=f"c2st_accuracy={accuracy:.3f}",
            )
        )

    overall = sum(m.accuracy for m in metrics) / len(metrics) if metrics else 0.5
    passed = overall <= threshold
    return DetectionReport(metrics=metrics, overall_score=overall, passed=passed)
