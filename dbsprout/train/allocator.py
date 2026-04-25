"""Stratified sample allocator: row counts -> per-table targets with min/max clamp."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dbsprout.train.models import SampleAllocation

if TYPE_CHECKING:
    from collections.abc import Mapping


def allocate_budget(
    *,
    row_counts: Mapping[str, int],
    budget: int,
    min_per_table: int,
    max_per_table: int,
) -> list[SampleAllocation]:
    """Allocate ``budget`` rows across tables proportional to row count.

    Targets are clamped to ``[min_per_table, max_per_table]``. The clamp on the
    table's own row count is implicit: a table is never asked for more rows
    than it actually has, so ``min_per_table`` is itself reduced to the table's
    row count when the table has fewer rows than the configured floor. After
    clamping, any rounding residual is redistributed by adjusting the
    largest unclamped tables one row at a time.

    Empty tables (row_count == 0) are excluded from the result.

    .. note::
        ``min_per_table`` is a **hard floor** by design: when
        ``min_per_table * n_non_empty_tables > budget`` the returned sum will
        legitimately overshoot ``budget`` (each non-empty table still receives
        at least ``min_per_table`` rows, capped at its own row count). This
        guarantees the floor is honored even at the cost of the budget bound.

    Raises:
        ValueError: if ``min_per_table > max_per_table``.
    """
    if min_per_table > max_per_table:
        raise ValueError(
            f"min_per_table ({min_per_table}) must be <= max_per_table ({max_per_table})"
        )

    non_empty = {t: c for t, c in row_counts.items() if c > 0}
    if not non_empty or budget <= 0:
        return [
            SampleAllocation(
                table=t,
                row_count=row_counts[t],
                target=0,
                floor_clamped=False,
                ceiling_clamped=False,
            )
            for t in non_empty
        ]

    total = sum(non_empty.values())
    raw = {t: budget * c / total for t, c in non_empty.items()}

    targets: dict[str, int] = {}
    for t, r in raw.items():
        bounded_max = min(max_per_table, non_empty[t])
        # Cap the floor at the table's row count: if the table has fewer rows
        # than ``min_per_table`` the row count itself becomes the binding cap
        # (neither user-configured floor nor ceiling is the binding bound).
        effective_min = min(min_per_table, non_empty[t])
        targets[t] = max(effective_min, min(bounded_max, round(r)))

    # Redistribute residual without crossing clamp bounds. We loop until either
    # the residual is zero or no table can absorb another row in the required
    # direction (everything is pinned at its bound). Each outer iteration that
    # finds a non-empty ``adjustables`` set reduces ``|residual|`` by exactly
    # one, so ``abs(residual) + 1`` iterations is a provably-sufficient cap.
    residual = budget - sum(targets.values())
    max_iters = abs(residual) + 1
    iterations = 0
    while residual != 0 and iterations < max_iters:
        iterations += 1
        adjustables = [
            t
            for t in non_empty
            if (residual > 0 and targets[t] < min(max_per_table, non_empty[t]))
            or (residual < 0 and targets[t] > min_per_table)
        ]
        if not adjustables:
            break
        # Adjust the largest-raw table first (most influential). The
        # ``adjustables`` filter already guarantees the step stays inside the
        # bound, so a single step per outer iteration is sufficient.
        adjustables.sort(key=lambda t: raw[t], reverse=True)
        t = adjustables[0]
        step = 1 if residual > 0 else -1
        targets[t] += step
        residual -= step

    # ``floor_clamped`` / ``ceiling_clamped`` reflect whether the user's
    # configured ``min_per_table`` / ``max_per_table`` was the binding bound.
    # Hitting the implicit row-count cap is not a "ceiling clamp" — it just
    # means the table is fully sampled.
    return [
        SampleAllocation(
            table=t,
            row_count=non_empty[t],
            target=targets[t],
            floor_clamped=targets[t] == min_per_table,
            ceiling_clamped=targets[t] == max_per_table,
        )
        for t in non_empty
    ]
