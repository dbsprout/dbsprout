from dbsprout.plugins.errors import PluginError, PluginValidationError


def test_plugin_error_is_exception():
    assert issubclass(PluginError, Exception)


def test_plugin_validation_error_subclasses_plugin_error():
    assert issubclass(PluginValidationError, PluginError)


def test_plugin_validation_error_carries_details():
    err = PluginValidationError(
        group="dbsprout.parsers", name="broken", reason="missing attr 'parse'"
    )
    assert err.group == "dbsprout.parsers"
    assert err.name == "broken"
    assert "parse" in str(err)
