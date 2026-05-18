"""Registry-first dispatch helpers for generation engines and output writers.

These helpers consult the plugin registry first; when the registry is empty
(editable installs without refreshed entry-point metadata, or test contexts
that monkeypatch ``entry_points`` to return nothing) they fall back to the
hard-wired in-tree mapping. This keeps the third-party plugin path live
without breaking dev / CI workflows that haven't run ``pip install -e .``.

Located in :mod:`dbsprout.plugins` (rather than ``dbsprout/cli``) so the
generation orchestrator can import these helpers without creating a
circular dependency on the CLI command modules.
"""

from __future__ import annotations

import inspect
from typing import Any

from dbsprout.plugins.errors import PluginError
from dbsprout.plugins.registry import get_registry


def _instantiate(obj: Any, group: str, name: str, **kwargs: Any) -> Any:
    """Call *obj* with *kwargs* if it is a class; convert constructor failures.

    Third-party plugin classes can blow up in their ``__init__`` (missing
    optional dep, bad signature, etc.). We surface those as
    :class:`PluginError` rather than letting an unwrapped traceback
    bubble out of CLI dispatch.
    """
    if not inspect.isclass(obj):
        return obj
    try:
        return obj(**kwargs)
    except Exception as exc:
        raise PluginError(
            f"Failed to instantiate plugin {group}:{name} ({type(exc).__name__}: {exc})"
        ) from exc


def resolve_writer(output_format: str) -> Any:
    """Return a writer instance for *output_format*.

    Entry points may register a class or a pre-built instance. When a
    class is returned (the common case from ``pyproject.toml``) it is
    instantiated with no arguments.
    """
    obj = get_registry().get("dbsprout.outputs", output_format)
    if obj is not None:
        return _instantiate(obj, "dbsprout.outputs", output_format)
    if output_format == "sql":
        from dbsprout.output.sql_writer import SQLWriter  # noqa: PLC0415

        return SQLWriter()
    if output_format == "csv":
        from dbsprout.output.csv_writer import CSVWriter  # noqa: PLC0415

        return CSVWriter()
    if output_format in ("json", "jsonl"):
        from dbsprout.output.json_writer import JSONWriter  # noqa: PLC0415

        return JSONWriter()
    if output_format == "parquet":
        from dbsprout.output.parquet_writer import ParquetWriter  # noqa: PLC0415

        return ParquetWriter()
    raise ValueError(f"Unknown output format: {output_format!r}")


def resolve_engine(engine: str, *, seed: int) -> Any:
    """Return a generation engine instance for *engine*.

    Entry points may register a class or a pre-built instance. When a
    class is returned it is instantiated with ``seed=seed`` (every
    in-tree engine accepts a ``seed`` kwarg).
    """
    obj = get_registry().get("dbsprout.generators", engine)
    if obj is not None:
        return _instantiate(obj, "dbsprout.generators", engine, seed=seed)
    if engine == "heuristic":
        from dbsprout.generate.engines.heuristic import HeuristicEngine  # noqa: PLC0415

        return HeuristicEngine(seed=seed)
    if engine == "spec_driven":
        from dbsprout.generate.engines.spec_driven import SpecDrivenEngine  # noqa: PLC0415

        return SpecDrivenEngine(seed=seed)
    if engine == "statistical":
        from dbsprout.generate.engines.statistical import (  # noqa: PLC0415
            StatisticalEngine,
        )

        return StatisticalEngine(seed=seed)
    raise ValueError(f"Unknown generation engine: {engine!r}")


__all__ = ["resolve_engine", "resolve_writer"]
