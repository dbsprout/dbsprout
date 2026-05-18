# Architecture Overview

DBSprout is a five-stage pipeline. Each stage is pluggable and the stages
do not import one another.

## Pipeline

```
SCHEMA INPUT --> SPEC GENERATION --> FK GRAPH --> DATA GENERATION --> VALIDATION --> OUTPUT
```

1. **Schema Input** — reads the schema via live DB introspection
   (SQLAlchemy for SQLite, PostgreSQL, MySQL, SQL Server) or file parsing
   (SQL DDL, DBML, Mermaid ERD, PlantUML, Prisma) into a unified
   `DatabaseSchema`.
2. **Spec Generation** — heuristic regex/fuzzy matching (100+ patterns) or
   an optional LLM analysis (one call per schema, cached) produces a
   `DataSpec`.
3. **FK Graph** — builds a dependency graph, detects cycles (Tarjan SCC),
   and resolves them via nullable-FK deferral.
4. **Data Generation** — Mimesis/NumPy value generation in topological
   order, with FK columns sampled from parent primary keys (never
   LLM-generated).
5. **Validation** — FK satisfaction, PK uniqueness, UNIQUE and NOT NULL
   checks; all must be 100%.
6. **Output** — SQL INSERT (3 dialects), CSV, JSON, JSONL, with
   insertion-order file numbering.

## Hybrid LLM-as-Architect

The LLM (when enabled) runs **once per schema** to produce a cached
`DataSpec`. All row generation is deterministic via Mimesis/NumPy, so the
same seed always reproduces the same data.

## Plugin System

Parsers, generators, output writers, LLM providers, and migration
frameworks are all discovered via Python entry points — every component is
replaceable.
