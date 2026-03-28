"""FK dependency graph, topological sort, and cycle detection.

Builds a directed acyclic graph of foreign key dependencies from a
``DatabaseSchema`` and computes a batched topological insertion order.
Each batch contains tables that can be generated in parallel.

When cycles exist, ``detect_cycles`` uses NetworkX SCC analysis to
identify all strongly connected components and candidate break edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from dbsprout.schema.models import ForeignKeySchema  # noqa: TC001 — Pydantic needs at runtime

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema, TableSchema


@dataclass(frozen=True)
class _DependencyData:
    """Intermediate result of FK analysis — shared by FKGraph and detect_cycles.

    Note: frozen=True prevents attribute reassignment but dict fields can
    still be mutated in-place. Inner values use frozenset/tuple for true
    immutability of contained data. This is a private helper — treat as
    read-only after construction.
    """

    deps: dict[str, frozenset[str]]
    self_refs: dict[str, tuple[ForeignKeySchema, ...]]
    ext_refs: dict[str, frozenset[str]]
    edges_by_table: dict[str, tuple[ForeignKeySchema, ...]]
    known_tables: frozenset[str] = field(default_factory=frozenset)


def _build_dependency_data(schema: DatabaseSchema) -> _DependencyData:
    """Extract FK relationships from a schema into a structured form.

    Separates self-referencing FKs, external refs (FK to table not in
    schema), and in-schema FK edges. Used by both ``FKGraph.from_schema``
    and ``detect_cycles``.
    """
    known_tables = frozenset(t.name for t in schema.tables)

    deps: dict[str, frozenset[str]] = {}
    self_refs: dict[str, tuple[ForeignKeySchema, ...]] = {}
    ext_refs: dict[str, frozenset[str]] = {}
    edges_by_table: dict[str, tuple[ForeignKeySchema, ...]] = {}

    for table in schema.tables:
        predecessors: set[str] = set()
        table_self_refs: list[ForeignKeySchema] = []
        table_ext_refs: set[str] = set()
        table_edges: list[ForeignKeySchema] = []

        for fk in table.foreign_keys:
            if fk.ref_table == table.name:
                table_self_refs.append(fk)
            elif fk.ref_table in known_tables:
                predecessors.add(fk.ref_table)
                table_edges.append(fk)
            else:
                table_ext_refs.add(fk.ref_table)

        deps[table.name] = frozenset(predecessors)

        if table_self_refs:
            self_refs[table.name] = tuple(table_self_refs)
        if table_ext_refs:
            ext_refs[table.name] = frozenset(table_ext_refs)
        if table_edges:
            edges_by_table[table.name] = tuple(table_edges)

    return _DependencyData(
        deps=deps,
        self_refs=self_refs,
        ext_refs=ext_refs,
        edges_by_table=edges_by_table,
        known_tables=known_tables,
    )


class FKGraph(BaseModel):
    """Immutable FK dependency graph with batched topological insertion order.

    Use the ``from_schema`` classmethod to construct from a
    ``DatabaseSchema``.  Raises ``graphlib.CycleError`` if the FK
    relationships contain cycles (S-007 handles cycle resolution).

    Table names within each batch of ``insertion_order`` are sorted
    alphabetically for deterministic, reproducible output regardless
    of input schema ordering.  The ``tables`` field is also sorted.
    """

    model_config = ConfigDict(frozen=True)

    # Note: dict fields are mutable containers on a frozen model. Pydantic's
    # frozen=True prevents attribute reassignment but not in-place dict mutation.
    # Treat all fields as read-only after construction. Inner values use
    # frozenset/tuple for true immutability of the contained data.
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

        data = _build_dependency_data(schema)
        order = _compute_insertion_order(data.deps)
        rev_deps = _compute_reverse_deps(data.deps)

        return cls(
            tables=tuple(sorted(data.known_tables)),
            dependencies=data.deps,
            self_referencing=data.self_refs,
            external_refs=data.ext_refs,
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


# ── Cycle detection ──────────────────────────────────────────────────────


class CycleEdge(BaseModel):
    """An FK edge within a cycle, tagged with its source table."""

    model_config = ConfigDict(frozen=True)

    source_table: str
    foreign_key: ForeignKeySchema


class CycleInfo(BaseModel):
    """Description of one strongly connected component (cycle) in the FK graph."""

    model_config = ConfigDict(frozen=True)

    tables: frozenset[str]
    edges: tuple[CycleEdge, ...] = ()
    candidate_breaks: tuple[CycleEdge, ...] = ()


def detect_cycles(schema: DatabaseSchema) -> tuple[CycleInfo, ...]:
    """Detect FK cycles using Tarjan's SCC algorithm.

    Returns an empty list for acyclic schemas.  Each ``CycleInfo``
    describes one strongly connected component with more than one
    table.  NetworkX is imported lazily — only when cycles exist.
    """
    if not schema.tables:
        return ()

    data = _build_dependency_data(schema)

    # Fast path: try topological sort — if it succeeds, no cycles
    try:
        _compute_insertion_order(data.deps)
        return ()
    except CycleError:
        pass

    # Slow path: find all SCCs via NetworkX
    import networkx as nx  # noqa: PLC0415

    graph: nx.DiGraph[str] = nx.DiGraph()
    for table_name, preds in data.deps.items():
        for pred in preds:
            graph.add_edge(pred, table_name)

    result: list[CycleInfo] = []
    for scc in nx.strongly_connected_components(graph):
        if len(scc) <= 1:
            continue

        edges: list[CycleEdge] = []
        candidates: list[CycleEdge] = []

        for table_name in sorted(scc):
            table_obj = schema.get_table(table_name)
            assert table_obj is not None  # SCC tables always exist in schema
            for fk in data.edges_by_table.get(table_name, ()):
                if fk.ref_table in scc:
                    edge = CycleEdge(source_table=table_name, foreign_key=fk)
                    edges.append(edge)
                    if _is_nullable_fk(fk, table_obj):
                        candidates.append(edge)

        def _edge_sort_key(e: CycleEdge) -> tuple[str, str, list[str]]:
            return (e.source_table, e.foreign_key.ref_table, e.foreign_key.columns)

        edges.sort(key=_edge_sort_key)
        candidates.sort(key=_edge_sort_key)

        result.append(
            CycleInfo(
                tables=frozenset(scc),
                edges=tuple(edges),
                candidate_breaks=tuple(candidates),
            )
        )

    result.sort(key=lambda c: tuple(sorted(c.tables)))
    return tuple(result)


def _is_nullable_fk(fk: ForeignKeySchema, table: TableSchema) -> bool:
    """True if ALL FK columns are nullable (candidate for cycle breaking)."""
    for c in fk.columns:
        col = table.get_column(c)
        if col is None or not col.nullable:
            return False
    return True


# ── Topological sort helpers ─────────────────────────────────────────────


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
