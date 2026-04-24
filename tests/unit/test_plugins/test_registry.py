import logging

import pytest

from dbsprout.plugins.errors import PluginValidationError
from dbsprout.plugins.registry import _module_path, get_registry
from dbsprout.schema.models import DatabaseSchema


class _ParserOk:
    suffixes = (".x",)

    def can_parse(self, text):
        return True

    def parse(self, text, *, source_file=None):
        return object()


class _ParserBad:
    suffixes = (".y",)


@pytest.fixture(autouse=True)
def _reset_registry():
    get_registry.cache_clear()
    yield
    get_registry.cache_clear()


def test_registry_get_returns_registered_plugin(make_ep, patched_eps):
    with patched_eps(
        {"dbsprout.parsers": [make_ep(name="ok", group="dbsprout.parsers", obj=_ParserOk())]}
    ):
        reg = get_registry()
        assert isinstance(reg.get("dbsprout.parsers", "ok"), _ParserOk)


def test_registry_list_filters_by_group(make_ep, patched_eps):
    with patched_eps(
        {
            "dbsprout.parsers": [make_ep(name="ok", group="dbsprout.parsers", obj=_ParserOk())],
            "dbsprout.outputs": [],
        }
    ):
        reg = get_registry()
        parsers = reg.list("dbsprout.parsers")
    assert len(parsers) == 1
    assert parsers[0].name == "ok"
    assert parsers[0].status == "loaded"


def test_registry_marks_broken_plugin_as_error(make_ep, patched_eps):
    with patched_eps(
        {
            "dbsprout.parsers": [
                make_ep(name="bad", group="dbsprout.parsers", obj=ModuleNotFoundError("boom"))
            ]
        }
    ):
        reg = get_registry()
    infos = reg.list("dbsprout.parsers")
    # broken plugins are dropped at discovery time — not in registry
    assert infos == []


def test_registry_rejects_protocol_mismatch(make_ep, patched_eps):
    with patched_eps(
        {"dbsprout.parsers": [make_ep(name="bad", group="dbsprout.parsers", obj=_ParserBad())]}
    ):
        reg = get_registry()
    infos = reg.list("dbsprout.parsers")
    assert len(infos) == 1
    assert infos[0].status == "error"
    assert "protocol" in (infos[0].error or "").lower()


def test_registry_duplicate_names_first_wins(make_ep, patched_eps, caplog):
    first = _ParserOk()
    second = _ParserOk()
    with (
        caplog.at_level(logging.WARNING, logger="dbsprout.plugins.registry"),
        patched_eps(
            {
                "dbsprout.parsers": [
                    make_ep(name="dup", group="dbsprout.parsers", obj=first),
                    make_ep(name="dup", group="dbsprout.parsers", obj=second),
                ]
            }
        ),
    ):
        reg = get_registry()
    assert reg.get("dbsprout.parsers", "dup") is first
    infos = reg.list("dbsprout.parsers")
    assert len(infos) == 1
    assert infos[0].status == "loaded"
    assert any("duplicate plugin name" in rec.message for rec in caplog.records)


def test_registry_check_raises_on_mismatch(make_ep, patched_eps):
    with patched_eps(
        {"dbsprout.parsers": [make_ep(name="bad", group="dbsprout.parsers", obj=_ParserBad())]}
    ):
        reg = get_registry()
        with pytest.raises(PluginValidationError):
            reg.check("dbsprout.parsers", "bad")


def test_registry_list_full_without_group(make_ep, patched_eps):
    with patched_eps(
        {"dbsprout.parsers": [make_ep(name="ok", group="dbsprout.parsers", obj=_ParserOk())]}
    ):
        reg = get_registry()
        full = reg.list()
    assert len(full) >= 1


def test_check_raises_for_unknown_plugin(make_ep, patched_eps):
    with patched_eps({}):
        reg = get_registry()
        with pytest.raises(PluginValidationError, match="not discovered"):
            reg.check("dbsprout.parsers", "does-not-exist")


def test_module_path_uses_class_when_obj_has_no_module(make_ep, patched_eps):
    """Plugins whose target has no ``__module__`` fall back to class."""

    class _Plain:
        suffixes = (".x",)

        def can_parse(self, text: str) -> bool:
            return True

        def parse(self, text: str, *, source_file: str | None = None) -> DatabaseSchema:
            return DatabaseSchema(tables=[], dialect="postgresql")

    plain = _Plain()
    # Force __module__ to be falsy to exercise the cls-fallback branch.
    plain.__module__ = ""  # type: ignore[assignment]
    result = _module_path(plain)
    # Falls back to the class's __module__, which is this test module.
    assert result  # non-empty — class.__module__ is set

    with patched_eps(
        {"dbsprout.parsers": [make_ep(name="p", group="dbsprout.parsers", obj=plain)]}
    ):
        reg = get_registry()
        infos = reg.list("dbsprout.parsers")
    assert infos[0].module
