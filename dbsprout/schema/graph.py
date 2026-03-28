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


class DeferredFK(BaseModel):
    """An FK edge deferred for two-pass insertion to break a cycle."""

    model_config = ConfigDict(frozen=True)

    source_table: str
    foreign_key: ForeignKeySchema
    reason: str = "cycle_break"


class ResolvedGraph(BaseModel):
    """Result of cycle resolution: an acyclic FKGraph + deferred FK metadata."""

    model_config = ConfigDict(frozen=True)

    graph: FKGraph
    deferred_fks: tuple[DeferredFK, ...] = ()


class UnresolvableCycleError(Exception):
    """Raised when a cycle has no nullable FK that can be deferred."""

    def __init__(self, cycle_info: CycleInfo) -> None:
        tables = ", ".join(sorted(cycle_info.tables))
        super().__init__(
            f"Cannot break cycle involving tables [{tables}]: "
            f"no nullable FK exists. Make one FK column nullable "
            f"or use DEFERRABLE constraints."
        )
        self.cycle_info = cycle_info


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
            if table_obj is None:  # pragma: no cover — SCC tables always exist in schema
                continue
            for fk in data.edges_by_table.get(table_name, ()):
                if fk.ref_table in scc:
                    edge = CycleEdge(source_table=table_name, foreign_key=fk)
                    edges.append(edge)
                    if _is_nullable_fk(fk, table_obj):
                        candidates.append(edge)

        def _edge_sort_key(e: CycleEdge) -> tuple[str, str, tuple[str, ...]]:
            return (e.source_table, e.foreign_key.ref_table, tuple(e.foreign_key.columns))

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


# ── Cycle breaking ───────────────────────────────────────────────────────

_MAX_BREAK_ITERATIONS = 10


def resolve_cycles(schema: DatabaseSchema) -> ResolvedGraph:
    """Resolve FK cycles and return an acyclic graph + deferred FK metadata.

    For acyclic schemas, returns the graph unchanged with no deferred FKs.
    For cyclic schemas, iteratively breaks cycles by deferring nullable FK
    edges until the graph becomes acyclic.

    Raises
    ------
    UnresolvableCycleError
        If a cycle has no nullable FK that can be deferred.
    """
    if not schema.tables:
        return ResolvedGraph(graph=FKGraph())

    data = _build_dependency_data(schema)

    # Fast path: try topological sort — if it succeeds, no cycles
    try:
        return ResolvedGraph(graph=_build_fk_graph(data))
    except CycleError:
        pass

    # Build table lookup for O(1) access
    table_lookup: dict[str, TableSchema] = {t.name: t for t in schema.tables}

    # Iteratively break cycles
    deferred: list[DeferredFK] = []
    modified_deps = dict(data.deps)  # mutable copy
    # Track deferred FK objects by identity (source_table, columns, ref_table)
    deferred_fk_ids: set[tuple[str, str, str]] = set()

    # First round: use full detect_cycles (returns CycleInfo with candidates)
    cycles = detect_cycles(schema)

    for _ in range(_MAX_BREAK_ITERATIONS):
        if not cycles:
            break

        for cycle_info in cycles:
            # Filter out already-deferred candidates
            remaining = tuple(
                c for c in cycle_info.candidate_breaks if _fk_id(c) not in deferred_fk_ids
            )
            if not remaining:
                raise UnresolvableCycleError(cycle_info)

            filtered_info = CycleInfo(
                tables=cycle_info.tables,
                edges=cycle_info.edges,
                candidate_breaks=remaining,
            )
            best = _pick_best_break(filtered_info)
            deferred.append(
                DeferredFK(
                    source_table=best.source_table,
                    foreign_key=best.foreign_key,
                )
            )
            deferred_fk_ids.add(_fk_id(best))

            # Update modified_deps: only remove ref_table from predecessor set
            # when ALL FKs from source to that ref_table are deferred
            src = best.source_table
            ref = best.foreign_key.ref_table
            all_fks_to_ref = [fk for fk in data.edges_by_table.get(src, ()) if fk.ref_table == ref]
            all_deferred = all(
                (src, ",".join(fk.columns), fk.ref_table) in deferred_fk_ids
                for fk in all_fks_to_ref
            )
            if all_deferred:
                modified_deps[src] = modified_deps[src] - {ref}

        # Check if all cycles are resolved
        try:
            _compute_insertion_order(modified_deps)
            break  # Acyclic now
        except CycleError:
            # Re-detect cycles from modified deps for next iteration
            cycles = _detect_cycles_from_deps(modified_deps, data, table_lookup)
    else:
        # Re-detect to get the remaining cycle info for the error
        remaining_cycles = _detect_cycles_from_deps(modified_deps, data, table_lookup)
        if remaining_cycles:
            raise UnresolvableCycleError(remaining_cycles[0])
        msg = "Could not resolve all cycles within iteration limit"  # pragma: no cover
        raise RuntimeError(msg)  # pragma: no cover

    # Build final graph from modified deps
    final_deps = {k: frozenset(v) for k, v in modified_deps.items()}
    order = _compute_insertion_order(final_deps)
    rev_deps = _compute_reverse_deps(final_deps)

    graph = FKGraph(
        tables=tuple(sorted(data.known_tables)),
        dependencies=final_deps,
        self_referencing=data.self_refs,
        external_refs=data.ext_refs,
        insertion_order=order,
        reverse_deps=rev_deps,
    )
    return ResolvedGraph(graph=graph, deferred_fks=tuple(deferred))


def _build_fk_graph(data: _DependencyData) -> FKGraph:
    """Build an FKGraph from pre-computed dependency data."""
    order = _compute_insertion_order(data.deps)
    rev_deps = _compute_reverse_deps(data.deps)
    return FKGraph(
        tables=tuple(sorted(data.known_tables)),
        dependencies=data.deps,
        self_referencing=data.self_refs,
        external_refs=data.ext_refs,
        insertion_order=order,
        reverse_deps=rev_deps,
    )


def _detect_cycles_from_deps(
    deps: dict[str, frozenset[str]],
    data: _DependencyData,
    table_lookup: dict[str, TableSchema],
) -> tuple[CycleInfo, ...]:
    """Re-detect cycles from a modified deps dict, returning full CycleInfo."""
    import networkx as nx  # noqa: PLC0415

    graph: nx.DiGraph[str] = nx.DiGraph()
    for table_name, preds in deps.items():
        for pred in preds:
            graph.add_edge(pred, table_name)

    result: list[CycleInfo] = []
    for scc in nx.strongly_connected_components(graph):
        if len(scc) <= 1:
            continue

        edges: list[CycleEdge] = []
        candidates: list[CycleEdge] = []

        for table_name in sorted(scc):
            table_obj = table_lookup.get(table_name)
            if table_obj is None:
                continue
            for fk in data.edges_by_table.get(table_name, ()):
                if fk.ref_table in scc and fk.ref_table in deps.get(table_name, frozenset()):
                    edge = CycleEdge(source_table=table_name, foreign_key=fk)
                    edges.append(edge)
                    if _is_nullable_fk(fk, table_obj):
                        candidates.append(edge)

        result.append(
            CycleInfo(
                tables=frozenset(scc),
                edges=tuple(edges),
                candidate_breaks=tuple(candidates),
            )
        )

    result.sort(key=lambda c: tuple(sorted(c.tables)))
    return tuple(result)


def _fk_id(edge: CycleEdge) -> tuple[str, str, str]:
    """Unique identifier for a CycleEdge: (source_table, columns_key, ref_table)."""
    return (edge.source_table, ",".join(edge.foreign_key.columns), edge.foreign_key.ref_table)


def _pick_best_break(cycle_info: CycleInfo) -> CycleEdge:
    """Pick the best nullable FK to defer from a cycle's candidates.

    Priority: deferrable FKs first, then table with most outgoing
    intra-SCC edges, then alphabetical for determinism.
    """
    candidates = cycle_info.candidate_breaks

    # Count outgoing intra-SCC edges per source table
    outgoing_counts: dict[str, int] = {}
    for edge in cycle_info.edges:
        outgoing_counts[edge.source_table] = outgoing_counts.get(edge.source_table, 0) + 1

    def sort_key(e: CycleEdge) -> tuple[int, int, str, str]:
        # Lower = better: deferrable first (0), then most outgoing (negative), then alpha
        is_deferrable = 0 if e.foreign_key.deferrable else 1
        outgoing = -outgoing_counts.get(e.source_table, 0)
        return (is_deferrable, outgoing, e.source_table, e.foreign_key.ref_table)

    return min(candidates, key=sort_key)


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
