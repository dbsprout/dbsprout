# DBSprout

Generate realistic seed data from your database schema.

DBSprout reads your database schema (live connection or schema file),
analyzes foreign key dependencies, and generates realistic seed data with
100% FK integrity. It works offline by default with an optional embedded AI
model, supports SQLite, PostgreSQL, MySQL, and SQL Server, and accepts
schemas from live databases, SQL DDL, DBML, Mermaid ERD, PlantUML, and
Prisma files.

## Highlights

- **Schema-first** — point at your database or a schema file, no config required.
- **7 schema inputs** — live DB or file (SQL DDL, DBML, Mermaid ERD, PlantUML, Prisma).
- **100% FK integrity** — topological ordering + FK sampling from parent PKs.
- **Deterministic** — same seed produces identical output for CI/testing.
- **Offline by default** — no internet, API key, or account required.

## Where to next

- [Install](getting-started/install.md) — get DBSprout onto your machine.
- [Quick Start](getting-started/quick-start.md) — install → init → generate in 5 minutes.
- [CLI Reference](user-guide/cli.md) — every command and option.
- [Configuration](user-guide/configuration.md) — the `dbsprout.toml` reference.
- [Architecture](architecture/overview.md) — the five-stage pipeline.
