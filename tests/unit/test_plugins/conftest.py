"""Shared test helpers for plugin tests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest import mock

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

import pytest


@dataclass
class FakeEntryPoint:
    """Minimal stand-in for ``importlib.metadata.EntryPoint``."""

    name: str
    group: str
    value: str
    loader: Callable[[], Any]

    def load(self) -> Any:
        return self.loader()


@pytest.fixture
def make_ep():
    def _make(*, name: str, group: str, obj: Any, value: str = "pkg:obj") -> FakeEntryPoint:
        def _loader() -> Any:
            if isinstance(obj, Exception):
                raise obj
            return obj

        return FakeEntryPoint(name=name, group=group, value=value, loader=_loader)

    return _make


@contextmanager
def _patch_entry_points(
    eps_by_group: dict[str, list[FakeEntryPoint]],
) -> Iterator[None]:
    def fake_entry_points(*, group: str | None = None, **_: Any):
        if group is None:
            return [ep for eps in eps_by_group.values() for ep in eps]
        return list(eps_by_group.get(group, []))

    with mock.patch("dbsprout.plugins.discovery.entry_points", side_effect=fake_entry_points):
        yield


@pytest.fixture
def patched_eps():
    return _patch_entry_points
