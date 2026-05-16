# Contributing

## Development setup

```bash
git clone https://github.com/dbsprout/dbsprout.git
cd dbsprout
uv sync --extra dev --extra data --extra stats
```

## Quality gates

```bash
uv run ruff check .                              # Lint
uv run ruff format --check .                     # Formatting
uv run mypy --strict dbsprout/                   # Type check
uv run bandit -c pyproject.toml -r dbsprout/     # Security scan
uv run pytest --cov=dbsprout                     # Tests (95% min coverage)
```

## Working on the docs site

```bash
uv sync --extra docs
uv run mkdocs serve            # Live preview at http://127.0.0.1:8000
uv run mkdocs build --strict   # Same check CI and the test gate run
```

Documentation source lives in `site_docs/`; `mkdocs.yml` is at the
repository root. The site deploys to GitHub Pages automatically on push to
`main`.
