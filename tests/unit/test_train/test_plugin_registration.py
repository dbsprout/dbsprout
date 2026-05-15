"""Verify train extractor + serializer satisfy their Protocols + entry points registered."""

from __future__ import annotations

from importlib.metadata import entry_points

from dbsprout.plugins.protocols import TrainExtractor, TrainSerializer
from dbsprout.train.extractor import SampleExtractor
from dbsprout.train.serializer import DataPreparer


def test_sample_extractor_satisfies_protocol() -> None:
    assert isinstance(SampleExtractor(), TrainExtractor)


def test_entry_point_registered() -> None:
    eps = entry_points(group="dbsprout.train_extractors")
    names = {ep.name for ep in eps}
    assert "live_db" in names


def test_data_preparer_satisfies_protocol() -> None:
    assert isinstance(DataPreparer(), TrainSerializer)


def test_serializer_entry_point_registered() -> None:
    eps = entry_points(group="dbsprout.train_serializers")
    names = {ep.name for ep in eps}
    assert "great" in names


def test_serializer_entry_point_resolves_to_class() -> None:
    eps = entry_points(group="dbsprout.train_serializers")
    great = next(ep for ep in eps if ep.name == "great")
    assert great.load() is DataPreparer
