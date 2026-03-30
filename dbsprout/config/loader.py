"""TOML configuration loader for DBSprout."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dbsprout.config.models import DBSproutConfig

if TYPE_CHECKING:
    from pathlib import Path

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover — Python 3.10
    import tomli as tomllib  # type: ignore[import-not-found]


def load_config(path: Path | None = None) -> DBSproutConfig:
    """Load and validate a ``dbsprout.toml`` configuration file.

    Parameters
    ----------
    path:
        Path to ``dbsprout.toml``. If ``None`` or the file does not
        exist, returns a ``DBSproutConfig`` with all defaults.

    Returns
    -------
    DBSproutConfig
        Validated, frozen configuration model.

    Raises
    ------
    pydantic.ValidationError
        If the TOML content contains invalid field types or unknown keys.
    """
    if path is None or not path.exists():
        return DBSproutConfig()

    try:
        with path.open("rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        msg = f"Failed to parse {path}: {exc}"
        raise ValueError(msg) from None

    return DBSproutConfig.model_validate(raw)
