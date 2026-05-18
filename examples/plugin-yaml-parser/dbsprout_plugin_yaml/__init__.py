"""Example dbsprout plugin: a YAML schema parser.

Registered via the ``dbsprout.parsers`` entry point in this package's
``pyproject.toml``. See the plugin development guide at
``site_docs/contributing/plugins.md``.
"""

from __future__ import annotations

from dbsprout_plugin_yaml.parser import YamlParser, build_schema, yaml_parser

__all__ = ["YamlParser", "build_schema", "yaml_parser"]
