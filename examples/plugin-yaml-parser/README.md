# dbsprout-plugin-yaml

Example [dbsprout](https://github.com/dbsprout/dbsprout) plugin: a YAML
schema parser. It demonstrates the dbsprout entry-point plugin API — see
the full guide in
[`site_docs/contributing/plugins.md`](../../site_docs/contributing/plugins.md).

## What it does

Registers a `SchemaParser` under the `dbsprout.parsers` entry-point group
that turns a small YAML schema document into dbsprout's `DatabaseSchema`.

## Install (editable)

```bash
pip install -e examples/plugin-yaml-parser
```

## Verify discovery

```bash
dbsprout plugins list
dbsprout plugins check dbsprout.parsers:yaml
```

## Use it

```bash
dbsprout init --file schema.yaml
```

## Run the example's tests

```bash
python -m pytest examples/plugin-yaml-parser/tests
```

## Example schema

```yaml
dialect: postgresql
tables:
  users:
    columns:
      id: {type: integer, primary_key: true}
      email: {type: varchar, unique: true, nullable: false}
    primary_key: [id]
```
