# CLI Reference

Run `dbsprout --help` for the live command list, or `dbsprout <command>
--help` for any command's options.

## `dbsprout init`

Introspect a database schema and generate configuration.

```
dbsprout init --db <connection-url>    # From a live database
dbsprout init --file <schema-file>     # From a schema file (.sql, .dbml, .mmd, .puml, .prisma)
dbsprout init --dry-run                # Preview without writing files
dbsprout init --output-dir ./config    # Custom output directory
```

## `dbsprout generate`

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

## `dbsprout validate`

Validate integrity of generated seed data.

```
dbsprout validate                              # Validate with defaults
dbsprout validate --rows 500                   # Validate 500 rows per table
dbsprout validate --engine spec                # Validate spec-driven output
dbsprout validate --format json                # JSON output (for CI pipelines)
```

## `dbsprout diff`

Report schema changes since the last snapshot.

```
dbsprout diff                                  # Show schema changes vs the stored snapshot
```

## `dbsprout audit`

Show the LLM interaction audit log.

```
dbsprout audit                                 # Show all audit entries
dbsprout audit --last 10                       # Show 10 most recent entries
```

## `dbsprout models`

List, download, and inspect embedded GGUF models.

```
dbsprout models                                # Inspect embedded GGUF models
```

## `dbsprout plugins`

Inspect discovered dbsprout plugins.

```
dbsprout plugins                               # List discovered plugins
```

## `dbsprout train`

Training pipeline subcommands (QLoRA fine-tuning).

```
dbsprout train --help                          # Show training subcommands
```
