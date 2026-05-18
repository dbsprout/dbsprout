# Quick Start

From zero to seed data in under five minutes.

## 1. Install

```bash
pip install dbsprout
```

## 2. Initialize from your database

```bash
# SQLite
dbsprout init --db sqlite:///myapp.db

# PostgreSQL
dbsprout init --db postgresql://user:pass@localhost:5432/mydb

# MySQL
dbsprout init --db mysql+pymysql://user:pass@localhost:3306/mydb
```

Or from a schema file:

```bash
dbsprout init --file schema.sql       # SQL DDL
dbsprout init --file schema.dbml      # DBML
dbsprout init --file schema.mmd       # Mermaid ERD
dbsprout init --file schema.puml      # PlantUML
dbsprout init --file schema.prisma    # Prisma
```

`init` introspects the schema, writes a `dbsprout.toml`, and stores a schema
snapshot under `.dbsprout/`.

## 3. Generate seed data

```bash
# Default heuristic engine: 100 rows/table, SQL INSERT, ./seeds/
dbsprout generate

# 500 rows per table as CSV
dbsprout generate --rows 500 --output-format csv

# Reproducible JSON output with an explicit seed
dbsprout generate --output-format json --seed 123
```

Output files are numbered by insertion order so they apply cleanly:

```
seeds/
  001_authors.sql
  002_books.sql
  003_orders.sql
```

## 4. Validate integrity

```bash
dbsprout validate
```

DBSprout validates FK satisfaction, PK uniqueness, UNIQUE and NOT NULL
constraints — all must be 100%.

!!! tip "Deterministic by default"
    The same `--seed` always produces identical output, which makes
    DBSprout safe to use inside CI pipelines and reproducible test fixtures.
