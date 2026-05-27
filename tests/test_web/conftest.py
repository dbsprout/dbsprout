"""Shared fixtures for the web test package.

The ``JobManager`` tests (S-108) are ``async`` and run via the ``anyio`` pytest
plugin (which ships transitively with ``starlette``/``fastapi`` in the ``[web]``
extra). ``anyio`` would otherwise parametrize every ``@pytest.mark.anyio`` test
across all installed backends; this fixture pins it to ``asyncio`` (the backend
the project uses; ``trio`` is not a dependency).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Pin anyio-driven async tests to the asyncio backend."""
    return "asyncio"
