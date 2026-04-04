# DBSprout

Generate realistic seed data from your database schema.

DBSprout reads your database schema (live connection or schema file), analyzes foreign key dependencies, and generates realistic seed data with 100% FK integrity. It works offline by default with an optional embedded AI model, supports SQLite, PostgreSQL, MySQL, and SQL Server, and accepts schemas from live databases, SQL DDL, DBML, Mermaid ERD, PlantUML, and Prisma files.

## Features

- **Schema-first** -- point at your DB or schema file, no config required
- **7 schema inputs** -- live DB (SQLite, PostgreSQL, MySQL, SQL Server) or file (SQL DDL, DBML, Mermaid ERD, PlantUML, Prisma)
- **100% FK integrity** -- topological ordering + FK sampling from parent PKs
- **Realistic values** -- 100+ pattern rules map columns to appropriate generators (email, name, phone, etc.)
- **AI-powered specs** -- optional embedded LLM (Qwen 2.5-1.5B) or cloud LLM (OpenAI, Anthropic, etc.) for smarter column mapping
- **Deterministic** -- same seed produces identical output for CI/testing
- **Multiple output formats** -- SQL INSERT, CSV, JSON, JSONL
- **3 SQL dialects** -- PostgreSQL, MySQL, SQLite with correct quoting and escaping
- **Constraint enforcement** -- UNIQUE dedup, NOT NULL, CHECK constraints (AC-3), auto-increment PKs
- **Geo coherence** -- city/state/zip tuples stay consistent across related columns
- **Integrity validation** -- automatic post-generation validation with detailed report
- **Cycle handling** -- detects and resolves circular FK dependencies automatically
- **Privacy tiers** -- Local, Redacted, and Cloud tiers control where data flows
- **PII redaction** -- Presidio-based detection and masking of sensitive data in LLM prompts
- **Audit logging** -- append-only JSON Lines log of all LLM interactions with cost tracking

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
pip install dbsprout[db]       # SQLAlchemy + drivers (PostgreSQL, MySQL)
pip install dbsprout[mssql]    # SQL Server support (pyodbc)
pip install dbsprout[llm]      # Embedded LLM (llama-cpp-python + Qwen 2.5-1.5B)
pip install dbsprout[cloud]    # Cloud LLM providers (LiteLLM + Instructor)
pip install dbsprout[privacy]  # PII redaction (Presidio)
pip install dbsprout[dev]      # Development tools (pytest, ruff, mypy)
```

## Quick Start

### 1. Initialize from your database

```bash
# SQLite
dbsprout init --db sqlite:///myapp.db

# PostgreSQL
dbsprout init --db postgresql://user:pass@localhost:5432/mydb

# MySQL
dbsprout init --db mysql+pymysql://user:pass@localhost:3306/mydb

# SQL Server
dbsprout init --db mssql+pyodbc://user:pass@localhost:1433/mydb?driver=ODBC+Driver+18+for+SQL+Server
```

Or from a schema file:

```bash
dbsprout init --file schema.sql       # SQL DDL
dbsprout init --file schema.dbml      # DBML
dbsprout init --file schema.mmd       # Mermaid ERD
dbsprout init --file schema.puml      # PlantUML
dbsprout init --file schema.prisma    # Prisma
```

### 2. Generate seed data

```bash
# Generate SQL INSERT files (default heuristic engine)
dbsprout generate

# Use AI-powered spec engine for smarter column mapping
dbsprout generate --engine spec

# Generate 500 rows per table with CSV output
dbsprout generate --rows 500 --output-format csv

# Generate JSON with a specific seed for reproducibility
dbsprout generate --output-format json --seed 123

# MySQL dialect
dbsprout generate --dialect mysql --output-dir ./mysql-seeds

# Cloud LLM with privacy tier (requires OPENAI_API_KEY or similar)
dbsprout generate --engine spec --privacy cloud
```

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
Done! Run `dbsprout generate` to create seed data.

$ dbsprout generate --rows 50 --output-format sql

         Integrity Validation
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Check            ┃ Table       ┃ Status ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━┩
│ pk_uniqueness    │ authors     │ PASS   │
│ pk_uniqueness    │ books       │ PASS   │
│ fk_satisfaction  │ books       │ PASS   │
│ fk_satisfaction  │ orders      │ PASS   │
│ fk_satisfaction  │ order_items │ PASS   │
└──────────────────┴─────────────┴────────┘
      Generation Complete
┏━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric     ┃ Value      ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Tables     │ 5          │
│ Total rows │ 250        │
│ Duration   │ 0.042s     │
│ Output     │ ./seeds    │
│ Format     │ sql        │
└────────────┴────────────┘
```

Output files are numbered by insertion order:

```
seeds/
  001_authors.sql
  001_categories.sql
  002_books.sql
  003_orders.sql
  004_order_items.sql
```

## CLI Reference

### `dbsprout init`

Introspect a database schema and generate configuration.

```
dbsprout init --db <connection-url>    # From live database
dbsprout init --file <schema-file>     # From schema file (.sql, .dbml, .mmd, .puml, .prisma)
dbsprout init --dry-run                # Preview without writing files
dbsprout init --output-dir ./config    # Custom output directory
```

### `dbsprout generate`

Generate seed data from the schema snapshot.

```
dbsprout generate                              # Defaults: 100 rows, SQL, ./seeds/
dbsprout generate --rows 500                   # 500 rows per table
dbsprout generate --seed 123                   # Deterministic output
dbsprout generate --output-format csv          # CSV output
dbsprout generate --output-format json         # JSON (pretty-printed array)
dbsprout generate --output-format jsonl        # JSONL (one object per line)
dbsprout generate --dialect mysql              # MySQL SQL dialect
dbsprout generate --output-dir ./my-seeds      # Custom output directory
dbsprout generate --engine spec                # AI-powered spec engine
dbsprout generate --privacy cloud              # Allow cloud LLM providers
dbsprout generate --schema-snapshot path.json  # Explicit schema path
dbsprout generate --config dbsprout.toml       # Explicit config path
```

### `dbsprout validate`

Validate integrity of generated seed data.

```
dbsprout validate                              # Validate with defaults
dbsprout validate --rows 500                   # Validate 500 rows per table
dbsprout validate --engine spec                # Validate spec-driven output
dbsprout validate --format json                # JSON output (for CI pipelines)
```

### `dbsprout audit`

Display the LLM interaction audit log.

```
dbsprout audit                                 # Show all audit entries
dbsprout audit --last 10                       # Show 10 most recent entries
```

## Configuration

`dbsprout.toml` (generated by `dbsprout init`):

```toml
[schema]
dialect = "postgresql"
source = "postgresql://user:***@localhost:5432/mydb"
snapshot = ".dbsprout/snapshots/a1b2c3d4.json"

[generation]
default_rows = 100
seed = 42
output_format = "sql"
output_dir = "./seeds"

# Per-table overrides
[tables.users]
rows = 50

[tables.audit_logs]
exclude = true
```

## How it Works

```
SCHEMA INPUT ──> SPEC GENERATION ──> FK GRAPH ──> DATA GENERATION ──> VALIDATION ──> OUTPUT
```

1. **Schema Input** -- Reads schema via live DB introspection (SQLAlchemy for 4 databases) or file parsing (SQL DDL, DBML, Mermaid, PlantUML, Prisma)
2. **Spec Generation** -- Heuristic regex matching (100+ patterns) or LLM-powered analysis (one call per schema, cached) produces a DataSpec
3. **FK Graph** -- Builds dependency graph, detects cycles (Tarjan SCC), resolves via nullable FK deferral
4. **Data Generation** -- Mimesis/NumPy value generation in topological order with FK sampling from parent PKs
5. **Constraints** -- UNIQUE dedup, NOT NULL, CHECK constraint enforcement (AC-3), auto-increment PKs, geo coherence
6. **Validation** -- FK satisfaction, PK uniqueness, UNIQUE, NOT NULL checks (all must be 100%)
7. **Output** -- SQL INSERT (3 dialects), CSV, JSON, JSONL with insertion-order file numbering

## Supported Databases

| Database   | Live Introspection | DDL File Parsing |
|------------|-------------------|------------------|
| SQLite     | Yes               | Yes              |
| PostgreSQL | Yes               | Yes              |
| MySQL      | Yes               | Yes              |
| SQL Server | Yes (`[mssql]`)   | Yes              |

## Supported Schema Formats

| Format   | Extension    | Notes                                |
|----------|-------------|--------------------------------------|
| SQL DDL  | `.sql`      | Auto-detects dialect via sqlglot     |
| DBML     | `.dbml`     | Full DBML spec via pydbml            |
| Mermaid  | `.mmd`      | erDiagram blocks with relationships  |
| PlantUML | `.puml`     | entity blocks with FK arrows         |
| Prisma   | `.prisma`   | Model definitions via DMMF extraction|

## Project Status

DBSprout is in active development. Sprints 1-4 complete (41 stories, 154 story points).

**Sprints 1-2** -- Core generation pipeline:
- Schema introspection for SQLite, PostgreSQL, MySQL
- SQL DDL file parsing with auto dialect detection
- FK dependency graph with topological sort and cycle resolution
- `dbsprout init` and `dbsprout generate` CLI commands
- Heuristic generation engine with 100+ column pattern rules
- Vectorized NumPy generation + deterministic seeding
- FK sampling, UNIQUE/NOT NULL constraint enforcement
- SQL INSERT (3 dialects), CSV, JSON/JSONL output writers
- Automatic integrity validation

**Sprint 3** -- AI-powered spec engine:
- Embedded LLM provider (Qwen 2.5-1.5B via llama-cpp-python)
- LLM spec analyzer with retry logic and spec caching
- Spec-driven generation engine (`--engine spec`)
- `dbsprout validate` command
- Geo coherence (563 US city/state/zip tuples)
- CHECK constraint enforcement (AC-3 arc consistency)

**Sprint 4** -- Schema everywhere + privacy:
- 4 new schema parsers: DBML, Mermaid ERD, PlantUML, Prisma
- Cloud LLM provider (OpenAI, Anthropic, etc. via LiteLLM + Instructor)
- Ollama provider for local LLM inference
- SQL Server introspection
- Privacy tier enforcement (Local / Redacted / Cloud)
- PII redaction via Presidio
- Append-only audit logging with `dbsprout audit` command

**1,400+ tests, 95%+ coverage**

**Coming next:** Direct database insertion (PostgreSQL COPY, MySQL LOAD DATA), Parquet output, UPSERT mode, quality metrics (fidelity, detection), and Django model introspection.

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

# Lint + type check + security scan
uv run ruff check .
uv run mypy --strict dbsprout/
uv run bandit -c pyproject.toml -r dbsprout/

# Run the CLI
uv run dbsprout --help
```

## License

MIT
