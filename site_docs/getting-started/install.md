# Install

**Requirements:** Python 3.10+

```bash
pip install dbsprout
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add dbsprout
```

## Optional extras

```bash
pip install dbsprout[db]       # SQLAlchemy + drivers (PostgreSQL, MySQL)
pip install dbsprout[mssql]    # SQL Server support (pyodbc)
pip install dbsprout[llm]      # Embedded LLM (llama-cpp-python + Qwen 2.5-1.5B)
pip install dbsprout[cloud]    # Cloud LLM providers (LiteLLM + Instructor)
pip install dbsprout[privacy]  # PII redaction (Presidio)
pip install dbsprout[dev]      # Development tools (pytest, ruff, mypy)
pip install dbsprout[docs]     # Documentation site tooling (mkdocs-material)
```

The core install is dependency-light; everything beyond schema parsing and
heuristic generation is an opt-in extra.
