"""Tests for the ``python -m dbsprout`` entry point (dbsprout.__main__)."""

from __future__ import annotations

import runpy
import sys
from unittest.mock import patch

import pytest

import dbsprout.__main__ as main_mod
import dbsprout.cli.app  # noqa: F401  (ensure submodule registered in sys.modules)
from dbsprout.cli.app import app

# ``dbsprout.cli.__init__`` re-exports ``app``, which shadows the
# ``dbsprout.cli.app`` submodule as a package attribute. On Python 3.10
# both ``import dbsprout.cli.app as x`` and ``mock.patch("dbsprout.cli.app.app")``
# resolve to the shadowing Typer object instead of the submodule (fixed in
# 3.11+). Fetch the submodule from ``sys.modules`` for a version-stable target.
cli_app_mod = sys.modules["dbsprout.cli.app"]


def test_main_module_exposes_app() -> None:
    """Importing the module exposes the Typer ``app`` object."""
    assert main_mod.app is app


def test_python_dash_m_invokes_app() -> None:
    """Running ``python -m dbsprout`` calls the Typer app."""
    sys.modules.pop("dbsprout.__main__", None)
    with (
        patch.object(cli_app_mod, "app") as mock_app,
        patch.object(sys, "argv", ["dbsprout", "--help"]),
    ):
        runpy.run_module("dbsprout", run_name="__main__")
    mock_app.assert_called_once_with()


def test_main_help_exits_zero() -> None:
    """The CLI entry point prints usage and exits cleanly for --help."""
    sys.modules.pop("dbsprout.__main__", None)
    with (
        patch.object(sys, "argv", ["dbsprout", "--help"]),
        pytest.raises(SystemExit) as exc,
    ):
        runpy.run_module("dbsprout", run_name="__main__")
    assert exc.value.code == 0
