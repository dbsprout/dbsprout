"""FK dependency graph and topological sort.

Builds a directed acyclic graph of foreign key dependencies from a
``DatabaseSchema`` and computes a batched topological insertion order.
Each batch contains tables that can be generated in parallel.
"""

from __future__ import annotations

from graphlib import TopologicalSorter
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from dbsprout.schema.models import ForeignKeySchema  # noqa: TC001 — Pydantic needs at runtime

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema


class FKGraph(BaseModel):
    """Immutable FK dependency graph with batched topological insertion order.

    Use the ``from_schema`` classmethod to construct from a
    ``DatabaseSchema``.  Raises ``graphlib.CycleError`` if the FK
    relationships contain cycles (S-007 handles cycle resolution).
    """

    model_config = ConfigDict(frozen=True)

    tables: tuple[str, ...] = ()
    dependencies: dict[str, frozenset[str]] = Field(default_factory=dict)
    self_referencing: dict[str, tuple[ForeignKeySchema, ...]] = Field(default_factory=dict)
    external_refs: dict[str, frozenset[str]] = Field(default_factory=dict)
    insertion_order: tuple[tuple[str, ...], ...] = ()
    reverse_deps: dict[str, frozenset[str]] = Field(default_factory=dict)

    @classmethod
    def from_schema(cls, schema: DatabaseSchema) -> FKGraph:
        """Build an FK graph from a ``DatabaseSchema``.

        Raises
        ------
        graphlib.CycleError
            If the FK relationships form a cycle.
        """
        if not schema.tables:
            return cls()

        known_tables = frozenset(t.name for t in schema.tables)

        deps: dict[str, frozenset[str]] = {}
        self_refs: dict[str, tuple[ForeignKeySchema, ...]] = {}
        ext_refs: dict[str, frozenset[str]] = {}

        for table in schema.tables:
            predecessors: set[str] = set()
            table_self_refs: list[ForeignKeySchema] = []
            table_ext_refs: set[str] = set()

            for fk in table.foreign_keys:
                if fk.ref_table == table.name:
                    table_self_refs.append(fk)
                elif fk.ref_table in known_tables:
                    predecessors.add(fk.ref_table)
                else:
                    table_ext_refs.add(fk.ref_table)

            deps[table.name] = frozenset(predecessors)

            if table_self_refs:
                self_refs[table.name] = tuple(table_self_refs)
            if table_ext_refs:
                ext_refs[table.name] = frozenset(table_ext_refs)

        order = _compute_insertion_order(deps)
        table_names = tuple(sorted(known_tables))
        rev_deps = _compute_reverse_deps(deps)

        return cls(
            tables=table_names,
            dependencies=deps,
            self_referencing=self_refs,
            external_refs=ext_refs,
            insertion_order=order,
            reverse_deps=rev_deps,
        )

    def dependents(self, table: str) -> frozenset[str]:
        """Return tables that depend on the given table (reverse edges).

        Raises
        ------
        KeyError
            If ``table`` is not in the graph.
        """
        if table not in self.dependencies:
            msg = f"Table {table!r} not in graph"
            raise KeyError(msg)
        return self.reverse_deps.get(table, frozenset())


def _compute_insertion_order(
    deps: dict[str, frozenset[str]],
) -> tuple[tuple[str, ...], ...]:
    """Run topological sort and return batched insertion order.

    Each batch is a tuple of table names sorted alphabetically
    for deterministic output.
    """
    sorter: TopologicalSorter[str] = TopologicalSorter(deps)
    sorter.prepare()

    batches: list[tuple[str, ...]] = []
    while sorter.is_active():
        ready = sorter.get_ready()
        batch = tuple(sorted(ready))
        batches.append(batch)
        sorter.done(*batch)

    return tuple(batches)


def _compute_reverse_deps(
    deps: dict[str, frozenset[str]],
) -> dict[str, frozenset[str]]:
    """Pre-compute reverse dependency map for O(1) dependents() lookups."""
    rev: dict[str, set[str]] = {}
    for table, predecessors in deps.items():
        for pred in predecessors:
            rev.setdefault(pred, set()).add(table)
    return {k: frozenset(v) for k, v in rev.items()}
