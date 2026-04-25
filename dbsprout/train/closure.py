"""FK closure pass: ensure every sampled child FK value has its parent row sampled too."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import polars as pl

from dbsprout.train.models import ClosureReport

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dbsprout.schema.models import DatabaseSchema, TableSchema

logger = logging.getLogger("dbsprout.train.closure")


@runtime_checkable
class ParentFetcher(Protocol):
    """Adapter shape: closure asks for parent rows by PK; the adapter does the SQL."""

    def fetch_by_pk(
        self,
        table: TableSchema,
        pk_column: str,
        values: Iterable[Any],
    ) -> pl.DataFrame: ...


def _missing_parent_values(
    samples: dict[str, pl.DataFrame],
    child_table: str,
    child_col: str,
    parent_table: str,
    parent_pk_col: str,
) -> set[Any]:
    """Set of FK values referenced by child rows but not yet present in parent samples."""
    child_values = set(samples[child_table][child_col].drop_nulls().to_list())
    parent_df = samples.get(parent_table)
    parent_values = set(parent_df[parent_pk_col].to_list()) if parent_df is not None else set()
    return child_values - parent_values


def _append_fetched(
    samples: dict[str, pl.DataFrame], parent_table_name: str, fetched: pl.DataFrame
) -> None:
    """Append fetched parent rows into the in-memory sample dict."""
    parent_df = samples.get(parent_table_name)
    samples[parent_table_name] = (
        pl.concat([parent_df, fetched]) if parent_df is not None else fetched
    )


def close_fk_graph(  # noqa: PLR0912 - validation/warning branches are inherent to the closure pass
    samples: dict[str, pl.DataFrame],
    schema: DatabaseSchema,
    engine: ParentFetcher,
    *,
    max_iterations: int,
) -> ClosureReport:
    """Iteratively pull missing parent rows until no FK is dangling or we hit the cap."""
    additions: dict[str, int] = defaultdict(int)
    unresolved: dict[str, int] = defaultdict(int)
    warned_no_pk: set[str] = set()
    warned_empty_parent: set[tuple[str, str]] = set()
    iterations = 0

    while iterations < max_iterations:
        added_this_pass = 0
        for table in schema.tables:
            if table.name not in samples:
                continue
            for fk in table.foreign_keys:
                parent_table = schema.get_table(fk.ref_table)
                if parent_table is None:
                    continue
                if not parent_table.primary_key:
                    if fk.ref_table not in warned_no_pk:
                        logger.warning(
                            "skipping FK closure for table '%s' (no primary key)",
                            fk.ref_table,
                        )
                        warned_no_pk.add(fk.ref_table)
                    continue
                parent_pk_col = fk.ref_columns[0]
                missing = _missing_parent_values(
                    samples, table.name, fk.columns[0], fk.ref_table, parent_pk_col
                )
                if not missing:
                    continue
                fetched = engine.fetch_by_pk(parent_table, parent_pk_col, missing)
                still_missing = missing - set(fetched[parent_pk_col].to_list())
                if still_missing:
                    unresolved[table.name] += len(still_missing)
                    key = (table.name, fk.ref_table)
                    if key not in warned_empty_parent:
                        logger.warning(
                            "parent '%s' is empty but child '%s' has non-NULL FK values; "
                            "sampled child rows will fail FK integrity downstream",
                            fk.ref_table,
                            table.name,
                        )
                        warned_empty_parent.add(key)
                if len(fetched) > 0:
                    _append_fetched(samples, fk.ref_table, fetched)
                    additions[fk.ref_table] += len(fetched)
                    added_this_pass += len(fetched)
        iterations += 1
        if added_this_pass == 0:
            break

    if iterations == max_iterations:
        logger.warning("closure terminated after %d iterations on FK cycle", iterations)

    return ClosureReport(
        iterations=iterations,
        additions=dict(additions),
        unresolved_per_table=dict(unresolved),
    )
