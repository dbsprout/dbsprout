import logging

from dbsprout.plugins.discovery import discover


def test_discover_yields_loaded_objects(make_ep, patched_eps):
    target = object()
    ep = make_ep(name="foo", group="dbsprout.parsers", obj=target)
    with patched_eps({"dbsprout.parsers": [ep]}):
        result = list(discover("dbsprout.parsers"))
    assert result == [("foo", target)]


def test_discover_skips_broken_plugin_and_logs_warning(make_ep, patched_eps, caplog):
    bad = make_ep(
        name="broken",
        group="dbsprout.parsers",
        obj=ModuleNotFoundError("no_such_pkg"),
    )
    with (
        caplog.at_level(logging.WARNING, logger="dbsprout.plugins.discovery"),
        patched_eps({"dbsprout.parsers": [bad]}),
    ):
        result = list(discover("dbsprout.parsers"))
    assert result == []
    assert any(
        "broken" in rec.message and "dbsprout.parsers" in rec.message for rec in caplog.records
    )


def test_discover_empty_group_returns_nothing(patched_eps):
    with patched_eps({}):
        assert list(discover("dbsprout.outputs")) == []
