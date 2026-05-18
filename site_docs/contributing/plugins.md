# Writing plugins

dbsprout is extensible through Python
[entry points](https://packaging.python.org/en/latest/specifications/entry-points/).
Every pipeline stage — schema parsing, spec generation, row generation,
output, migration detection, and training extraction — can be replaced
or extended by a third-party package, with no changes to dbsprout
itself.

## How discovery works

On first use dbsprout builds a process-wide registry by enumerating the
six entry-point groups below. For each entry point it imports the target
object and runs an `isinstance` check against the group's
`@runtime_checkable` Protocol.

Because that check inspects **class-level** attributes, declare Protocol
data attributes (`suffixes`, `format`) on the class, not inside
`__init__`. Plugins that fail the Protocol check are recorded with
`status="error"` and skipped at dispatch; entry points that fail to
import are logged and dropped. Inspect the result with:

```bash
dbsprout plugins list
dbsprout plugins check dbsprout.parsers:yaml
```

!!! note "Register an instance for parsers"
    `dbsprout.parsers` dispatch (`parse_schema_file`) calls `can_parse`
    and `parse` directly on the registered object without instantiating
    it, and the registry's `isinstance` check needs a real instance.
    Point the parser entry point at a **module-level instance** (the
    in-tree adapters do this), e.g. `pkg:yaml_parser`, not the class
    `pkg:YamlParser`. Generator and output entry points, by contrast,
    are instantiated by the dispatcher, so they point at the class.

## Entry-point groups and Protocols

| Group | Protocol | Required interface |
|-------|----------|--------------------|
| `dbsprout.parsers` | `SchemaParser` | `suffixes: tuple[str, ...]`; `can_parse(text: str) -> bool`; `parse(text: str, *, source_file: str \| None = None) -> DatabaseSchema` |
| `dbsprout.generators` | `GenerationEngine` | `generate_table(*args, **kwargs) -> list[dict[str, Any]]` |
| `dbsprout.outputs` | `OutputWriter` | `format: str`; `write(*args, **kwargs) -> Any` |
| `dbsprout.llm_providers` | `SpecProvider` | `generate_spec(schema: DatabaseSchema) -> DataSpec` |
| `dbsprout.migration_frameworks` | `MigrationParser` | `detect_changes(project_path: Path) -> list[SchemaChange]` |
| `dbsprout.train_extractors` | `TrainExtractor` | `extract(*, source: str, config: ExtractorConfig) -> SampleResult` |

### SchemaParser

`suffixes` is the tuple of file extensions the parser claims (e.g.
`(".yaml", ".yml")`). `can_parse(text)` returns whether the parser can
handle a given file's contents — return `False` (never raise) so
dispatch can fall through to another parser. `parse(text, *,
source_file=None)` returns a populated `DatabaseSchema`; raise
`ValueError` on malformed input.

### GenerationEngine

`generate_table` returns a list of row dicts for one table. The concrete
signature varies across in-tree engines (some take a column→generator
mapping, others a per-table `DataSpec`); the Protocol only fixes the
method name and return type, so callers pass the kwargs the engine
documents. Engine entry points point at the class — the dispatcher
instantiates it with `seed=...`.

### OutputWriter

`format` is the format key users select (e.g. `"csv"`). `write` persists
generated rows for a full schema; accept the kwargs you need and ignore
the rest. Output entry points point at the class — the dispatcher
instantiates it with no arguments.

### SpecProvider

`generate_spec(schema)` returns a `DataSpec` describing per-column
generators, distributions, and correlations. Providers run once per
schema and the result is cached.

### MigrationParser

`detect_changes(project_path)` inspects a migration framework's project
directory and returns the list of `SchemaChange` objects describing what
changed since the last seed.

### TrainExtractor

`extract(*, source, config)` pulls a stratified `SampleResult` from a
data source for fine-tuning. `source` is a plain string (URL, path, …),
so new extractors (CSV dump, Parquet dir, …) need no new Protocol.

## Example: a YAML schema parser

A complete, runnable example lives in
[`examples/plugin-yaml-parser/`](https://github.com/dbsprout/dbsprout/tree/main/examples/plugin-yaml-parser).
It registers a `SchemaParser` for `.yaml`/`.yml` files.

### 1. Declare the entry point

`pyproject.toml`:

```toml
[project]
name = "dbsprout-plugin-yaml"
dependencies = ["dbsprout", "pyyaml>=6.0"]

[project.entry-points."dbsprout.parsers"]
yaml = "dbsprout_plugin_yaml:yaml_parser"
```

The key (`yaml`) is the plugin name shown by `dbsprout plugins list`;
the value is `module:attribute` and resolves to a module-level
`YamlParser()` instance.

### 2. Implement the Protocol

```python
class YamlParser:
    suffixes: tuple[str, ...] = (".yaml", ".yml")

    def can_parse(self, text: str) -> bool:
        ...  # True only when text is a YAML schema mapping

    def parse(
        self, text: str, *, source_file: str | None = None
    ) -> DatabaseSchema:
        ...  # build and return a DatabaseSchema


yaml_parser = YamlParser()  # the entry point points here
```

`suffixes` is class-level so the registry's `isinstance` check passes.

### 3. Test it

The example ships unit tests
(`examples/plugin-yaml-parser/tests/test_parser.py`) and dbsprout's own
test suite loads it through the real registry and drives it end-to-end
via `parse_schema_file`
(`tests/integration/test_plugins/test_example_yaml_plugin.py`).

### 4. Install and verify

```bash
pip install -e examples/plugin-yaml-parser
dbsprout plugins list
dbsprout plugins check dbsprout.parsers:yaml
dbsprout init --file schema.yaml
```

Once installed the parser is auto-discovered: any `.yaml`/`.yml` schema
file passed to `dbsprout init` / `generate` / `diff` is routed to it.
