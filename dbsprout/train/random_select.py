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
_PG_OVERSAMPLE_FACTOR: Final[int] = 3


@dataclass(frozen=True)
class RandomQuery:
    """Bound SQL + parameters for a dialect-specific random sample."""

    sql: str
    params: dict[str, object]
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
        sql = f'SELECT setseed(:s); SELECT * FROM "{name}" ORDER BY random() LIMIT :n'  # noqa: S608  # nosec B608 - name from SA reflection
        return RandomQuery(sql=sql, params={"s": seed / (2**31 - 1), "n": n})

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
        # Seed-derived linear-congruential coefficients.
        a = (seed * 6364136223846793005 + 1) % (2**31 - 1) or 1
        b = (seed * 1442695040888963407 + 17) % (2**31 - 1)
        p = 2**31 - 1
        sql = f'SELECT * FROM "{name}" ORDER BY ((rowid * :a + :b) % :p) LIMIT :n'  # noqa: S608  # nosec B608 - name from SA reflection
        return RandomQuery(sql=sql, params={"a": a, "b": b, "p": p, "n": n})

    raise ValueError(f"unsupported dialect: {dialect}")
