"""Dialect-aware random sampling query builder.

Returns a `RandomQuery` (raw SQL string + bound parameters) so the caller can
execute via `engine.connect().execute(text(sql), params)`. Table name is taken
from a SQLAlchemy `Table` object (already validated by reflection); never from
user input. All values are bound parameters - no f-string SQL.

Bandit ``S608`` warnings on the f-string SQL below are false positives: the
only interpolated value is ``table.name`` which comes from SQLAlchemy
``Inspector`` reflection, not user input. Each occurrence is marked
``# noqa: S608``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import sqlalchemy as sa

_PG_TABLESAMPLE_THRESHOLD: Final[int] = 1_000_000
# Oversample so BERNOULLI's per-row variance still yields >= n rows w.h.p.
_PG_OVERSAMPLE_FACTOR: Final[int] = 3


@dataclass(frozen=True)
class RandomQuery:
    """Dialect-specific random query.

    `params` keys vary per dialect (e.g. `n`, `seed`, `p`, `a`, `b`); callers
    should pass the whole dict to `text(sql).bindparams(**params)`.
    `setup` is a tuple of (sql, params) pairs to execute in the SAME
    transaction immediately before `sql` (e.g. `setseed` on PostgreSQL).
    """

    sql: str
    params: dict[str, object]
    setup: tuple[tuple[str, dict[str, object]], ...] = ()
    warning: str | None = None


def build_random_query(  # noqa: PLR0913 - dialect/seed/row_count/has_rowid each carry distinct meaning
    table: sa.Table,
    n: int,
    *,
    dialect: str,
    seed: int,
    row_count: int,
    has_rowid: bool = True,
) -> RandomQuery:
    """Build a `RandomQuery` for the given dialect."""
    name = table.name  # safe: came from SQLAlchemy reflection
    if dialect == "postgresql":
        if row_count >= _PG_TABLESAMPLE_THRESHOLD:
            pct = min(100.0, 100.0 * _PG_OVERSAMPLE_FACTOR * n / row_count)
            sql = (
                f'SELECT * FROM "{name}" '  # noqa: S608  # nosec B608 - name from SA reflection
                f"TABLESAMPLE BERNOULLI(:p) REPEATABLE (:seed) LIMIT :n"
            )
            return RandomQuery(sql=sql, params={"p": pct, "seed": seed, "n": n})
        s = max(-1.0, min(1.0, seed / (2**31 - 1)))
        sql = f'SELECT * FROM "{name}" ORDER BY random() LIMIT :n'  # noqa: S608  # nosec B608 - name from SA reflection
        return RandomQuery(
            sql=sql,
            params={"n": n},
            setup=(("SELECT setseed(:s)", {"s": s}),),
        )

    if dialect == "mysql":
        sql = f"SELECT * FROM `{name}` ORDER BY RAND(:seed) LIMIT :n"  # noqa: S608  # nosec B608 - name from SA reflection
        return RandomQuery(sql=sql, params={"seed": seed, "n": n})

    if dialect == "sqlite":
        if not has_rowid:
            sql = f'SELECT * FROM "{name}" ORDER BY random() LIMIT :n'  # noqa: S608  # nosec B608 - name from SA reflection
            return RandomQuery(
                sql=sql,
                params={"n": n},
                warning=f"seed ignored for WITHOUT ROWID table '{name}'",
            )
        # Seed-derived linear-congruential coefficients. Pre-mix the raw seed
        # with the SplittableRandom golden gamma so seed=0 (the default) doesn't
        # collapse `a` to 1 — which would degenerate the ORDER BY into the
        # monotonic ``rowid`` order and silently destroy the random sample.
        mixed_seed = seed ^ 0x9E3779B97F4A7C15
        a = (mixed_seed * 6364136223846793005 + 1) % (2**31 - 1) or 1
        b = (mixed_seed * 1442695040888963407 + 17) % (2**31 - 1)
        p = 2**31 - 1
        sql = f'SELECT * FROM "{name}" ORDER BY ((rowid * :a + :b) % :p) LIMIT :n'  # noqa: S608  # nosec B608 - name from SA reflection
        return RandomQuery(sql=sql, params={"a": a, "b": b, "p": p, "n": n})

    raise ValueError(f"unsupported dialect: {dialect}")
