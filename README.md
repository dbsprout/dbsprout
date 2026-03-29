# DBSprout

Generate realistic seed data from your database schema.

DBSprout reads your database schema (live connection or DDL file), analyzes foreign key dependencies, resolves cycles, and generates seed data in the correct insertion order. It works offline by default and supports SQLite, PostgreSQL, and MySQL.

## Installation

```bash
pip install dbsprout
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add dbsprout
```

**Requirements:** Python 3.10+

### Optional extras

```bash
pip install dbsprout[db]      # SQLAlchemy + database drivers (psycopg2, pymysql)
pip install dbsprout[dev]     # Development tools (pytest, ruff, mypy)
```

## Quick Start

### From a live database

Point DBSprout at your database and it will introspect the schema, display a summary, and generate a configuration file:

```bash
# SQLite
dbsprout init --db sqlite:///myapp.db

# PostgreSQL
dbsprout init --db postgresql://user:pass@localhost:5432/mydb

# MySQL
dbsprout init --db mysql+pymysql://user:pass@localhost:3306/mydb
```

### From a DDL file

No running database? Use a SQL DDL file instead:

```bash
dbsprout init --file schema.sql
```

DBSprout auto-detects the SQL dialect (PostgreSQL, MySQL, SQLite) from the DDL content.

### Example output

```
$ dbsprout init --db sqlite:///bookstore.db

            Schema Summary
┏━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━┳━━━━━━━━━━━━━┓
┃ Table       ┃ Columns ┃ FKs ┃ Primary Key ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━╇━━━━━━━━━━━━━┩
│ authors     │       3 │   0 │ id          │
│ books       │       4 │   1 │ id          │
│ categories  │       3 │   1 │ id          │
│ orders      │       4 │   1 │ id          │
│ order_items │       4 │   2 │ id          │
└─────────────┴─────────┴─────┴─────────────┘
╭──────────────── Insertion Order ─────────────────╮
│   1. authors, categories                         │
│   2. books                                       │
│   3. orders                                      │
│   4. order_items                                 │
╰──────────────────────────────────────────────────╯
Self-referencing FKs: categories

Wrote dbsprout.toml
Wrote .dbsprout/snapshots/a1b2c3d4e5f6g7h8.json

Done! Run `dbsprout generate` to create seed data.
```

### Preview without writing files

```bash
dbsprout init --db sqlite:///myapp.db --dry-run
```

### Specify output directory

```bash
dbsprout init --db sqlite:///myapp.db --output-dir ./config
```

## What it generates

### `dbsprout.toml`

Configuration file with your schema settings and generation defaults:

```toml
# DBSprout configuration

[schema]
dialect = "sqlite"
source = "sqlite:///myapp.db"
snapshot = ".dbsprout/snapshots/a1b2c3d4.json"

[generation]
default_rows = 100
seed = 42
output_format = "sql"
output_dir = "./seeds"

# Per-table overrides (uncomment and customize):
# [tables.users]
# rows = 50
# [tables.orders]
# rows = 200
```

### Schema snapshot

A JSON snapshot of your schema is saved to `.dbsprout/snapshots/` for future diff operations. The filename is a hash of the schema content, so identical schemas produce the same file.

## How it works

1. **Schema Input** -- Reads your schema via live database introspection (SQLAlchemy) or DDL file parsing (sqlglot)
2. **FK Graph Analysis** -- Builds a dependency graph from foreign key relationships
3. **Cycle Detection** -- Uses Tarjan's SCC algorithm to find circular dependencies
4. **Cycle Resolution** -- Automatically breaks cycles by deferring nullable FK columns for two-pass insertion
5. **Insertion Order** -- Computes a batched topological sort so parent tables are populated before children

## Supported databases

| Database   | Live introspection | DDL file parsing |
|------------|-------------------|------------------|
| SQLite     | Yes               | Yes              |
| PostgreSQL | Yes               | Yes              |
| MySQL      | Yes               | Yes              |

## Project status

DBSprout is in active development. Sprint 1 (schema input layer) is complete:

- Schema introspection for SQLite, PostgreSQL, MySQL
- SQL DDL file parsing with auto dialect detection
- FK dependency graph with topological sort
- Cycle detection (Tarjan SCC) and automatic cycle breaking
- `dbsprout init` CLI command with Rich terminal output
- 375+ tests, 95%+ coverage

**Coming next:** Seed data generation engines, output writers, and the `dbsprout generate` command.

## Development

```bash
# Clone and install
git clone https://github.com/dbsprout/dbsprout.git
cd dbsprout
uv sync --extra dev

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=dbsprout --cov-report=term-missing

# Lint + type check
uv run ruff check .
uv run mypy --strict dbsprout/

# Run the CLI
uv run dbsprout --help
```

## License

MIT
