import pytest
from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.plugins.registry import get_registry


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


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_plugins_list_shows_loaded(make_ep, patched_eps, runner):
    with patched_eps(
        {"dbsprout.parsers": [make_ep(name="ok", group="dbsprout.parsers", obj=_ParserOk())]}
    ):
        result = runner.invoke(app, ["plugins", "list"])
    assert result.exit_code == 0
    assert "dbsprout.parsers" in result.stdout
    assert "ok" in result.stdout
    assert "loaded" in result.stdout


def test_plugins_check_ok_exits_zero(make_ep, patched_eps, runner):
    with patched_eps(
        {"dbsprout.parsers": [make_ep(name="ok", group="dbsprout.parsers", obj=_ParserOk())]}
    ):
        result = runner.invoke(app, ["plugins", "check", "dbsprout.parsers:ok"])
    assert result.exit_code == 0


def test_plugins_check_bad_exits_two(make_ep, patched_eps, runner):
    with patched_eps(
        {"dbsprout.parsers": [make_ep(name="bad", group="dbsprout.parsers", obj=_ParserBad())]}
    ):
        result = runner.invoke(app, ["plugins", "check", "dbsprout.parsers:bad"])
    assert result.exit_code == 2
    assert "Protocol" in result.stdout or "does not satisfy" in result.stdout


def test_plugins_check_bad_target_format_exits_two(runner):
    result = runner.invoke(app, ["plugins", "check", "missing-colon"])
    assert result.exit_code == 2
