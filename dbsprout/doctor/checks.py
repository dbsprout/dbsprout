"""Pure environment-health check functions for ``dbsprout doctor``.

Each public check returns a :class:`CheckResult` and performs no console
I/O. ``run_all_checks`` orchestrates every check and never raises.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Status = Literal["pass", "warn", "fail"]

_MIN_PY = (3, 10)
_MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
_MIN_FREE_BYTES = 1024**3  # 1 GiB

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai-style key", re.compile(r"sk-[A-Za-z0-9]{16,}")),
    ("aws access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    (
        "inline credential",
        re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*=\s*['\"][^'\"]{8,}"),
    ),
)

_EXTRA_MODULES: tuple[tuple[str, str], ...] = (
    ("sqlalchemy", "db"),
    ("mimesis", "gen"),
    ("numpy", "gen"),
    ("llama_cpp", "llm"),
    ("huggingface_hub", "llm"),
    ("scipy", "stats"),
    ("sklearn", "stats"),
    ("polars", "data"),
)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single environment check."""

    category: str
    name: str
    status: Status
    message: str
    fix: str | None = None


def check_python_version(
    version_info: tuple[int, int, int] | None = None,
) -> CheckResult:
    """Verify the interpreter is Python 3.10 or newer."""
    vi = version_info if version_info is not None else sys.version_info[:3]
    found = ".".join(str(p) for p in vi)
    if vi[:2] >= _MIN_PY:
        return CheckResult("Environment", "python", "pass", f"Python {found} (>= 3.10)")
    return CheckResult(
        "Environment",
        "python",
        "fail",
        f"Python {found} is too old; 3.10+ required",
        fix="Install Python 3.10 or newer and recreate the virtualenv.",
    )


def _module_version(module: str) -> str:
    dist = {"sklearn": "scikit-learn", "llama_cpp": "llama-cpp-python"}.get(module, module)
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def check_extras() -> list[CheckResult]:
    """Report which optional dependency modules are importable."""
    results: list[CheckResult] = []
    for module, extra in _EXTRA_MODULES:
        present = importlib.util.find_spec(module) is not None
        if present:
            results.append(
                CheckResult(
                    "Environment",
                    f"extra:{module}",
                    "pass",
                    f"{module} {_module_version(module)} installed",
                )
            )
        else:
            results.append(
                CheckResult(
                    "Environment",
                    f"extra:{module}",
                    "warn",
                    f"{module} not installed (optional)",
                    fix=f"Install with: pip install dbsprout[{extra}]",
                )
            )
    return results


def check_database(db_url: str | None) -> CheckResult:
    """Attempt ``SELECT 1`` against a configured database URL."""
    if not db_url:
        return CheckResult(
            "Database",
            "connectivity",
            "pass",
            "No database configured (skipped)",
        )
    import sqlalchemy as sa  # noqa: PLC0415

    from dbsprout.schema.introspect import _create_engine  # noqa: PLC0415

    try:
        safe = sa.engine.make_url(db_url).render_as_string(hide_password=True)
    except Exception:
        safe = "<unparseable url>"
    try:
        engine = _create_engine(db_url)
        try:
            with engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
        finally:
            engine.dispose()
    except Exception as exc:
        return CheckResult(
            "Database",
            "connectivity",
            "fail",
            f"Could not connect to {safe}: {type(exc).__name__}",
            fix="Verify the DB URL/credentials and that the server is reachable.",
        )
    return CheckResult("Database", "connectivity", "pass", f"Connected to {safe}")


def _default_model_root() -> Path:
    return Path.home() / ".cache" / "dbsprout" / "models"


def check_model(model_root: Path | None = None) -> CheckResult:
    """Report whether the embedded GGUF model has been downloaded."""
    root = model_root if model_root is not None else _default_model_root()
    found = root.exists() and any(root.rglob(_MODEL_FILE))
    if found:
        return CheckResult("Models", "embedded", "pass", f"Embedded model present in {root}")
    return CheckResult(
        "Models",
        "embedded",
        "warn",
        f"Embedded model not downloaded (looked in {root})",
        fix="Run an LLM-engine command (e.g. dbsprout generate --engine spec) "
        "to fetch it, or `pip install dbsprout[llm]`.",
    )


def _existing_ancestor(path: Path) -> Path:
    p = path.resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def check_disk_space(dbsprout_parent: Path | None = None) -> CheckResult:
    """Warn when free space near ``.dbsprout/`` is below 1 GiB."""
    target = _existing_ancestor((dbsprout_parent or Path.cwd()) / ".dbsprout")
    free = shutil.disk_usage(target).free
    gib = free / 1024**3
    if free >= _MIN_FREE_BYTES:
        return CheckResult("Environment", "disk", "pass", f"{gib:.1f} GiB free at {target}")
    return CheckResult(
        "Environment",
        "disk",
        "warn",
        f"Only {gib:.2f} GiB free at {target}",
        fix="Free up disk space before downloading models or large seeds.",
    )


def check_plugins() -> CheckResult:
    """Report discovered plugins and any that failed to load."""
    from dbsprout.plugins.registry import get_registry  # noqa: PLC0415

    infos = get_registry().list()
    errored = [i for i in infos if i.status == "error"]
    if errored:
        names = ", ".join(f"{i.group}:{i.name}" for i in errored)
        return CheckResult(
            "Plugins",
            "registry",
            "warn",
            f"{len(infos)} discovered, {len(errored)} failed: {names}",
            fix="Run `dbsprout plugins check <group>:<name>` for details.",
        )
    return CheckResult(
        "Plugins",
        "registry",
        "pass",
        f"{len(infos)} plugin(s) discovered, all loaded",
    )


def check_secrets(config_path: Path | None) -> CheckResult:
    """Warn when API-key-like patterns appear in a tracked config file."""
    if config_path is None or not config_path.exists():
        return CheckResult("Privacy", "secrets", "pass", "No config file to scan")
    text = config_path.read_text(encoding="utf-8", errors="replace")
    hits: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for label, pattern in _SECRET_PATTERNS:
            if pattern.search(line):
                hits.append(f"{label} (line {lineno})")
    if not hits:
        return CheckResult(
            "Privacy",
            "secrets",
            "pass",
            f"No secret-like patterns in {config_path.name}",
        )
    return CheckResult(
        "Privacy",
        "secrets",
        "warn",
        f"Possible secrets in {config_path.name}: {'; '.join(hits)}",
        fix="Move secrets out of the config file into environment variables.",
    )


def run_all_checks(
    *,
    db_url: str | None = None,  # noqa: ARG001 - wired up in Task 8
    config_path: Path | None = None,  # noqa: ARG001 - wired up in Task 8
) -> list[CheckResult]:
    """Placeholder — full orchestration implemented in Task 8."""
    return []
