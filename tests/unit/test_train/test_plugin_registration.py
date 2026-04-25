"""Verify SampleExtractor satisfies the TrainExtractor Protocol + entry point registered."""

from __future__ import annotations

from importlib.metadata import entry_points

from dbsprout.plugins.protocols import TrainExtractor
from dbsprout.train.extractor import SampleExtractor


def test_sample_extractor_satisfies_protocol() -> None:
    assert isinstance(SampleExtractor(), TrainExtractor)


def test_entry_point_registered() -> None:
    eps = entry_points(group="dbsprout.train_extractors")
    names = {ep.name for ep in eps}
    assert "live_db" in names
