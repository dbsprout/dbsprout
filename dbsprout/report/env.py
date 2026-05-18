"""Jinja2 environment + render entry point for the HTML report.

Templates ship inside the installed package (``dbsprout/report/templates``)
and are located via :func:`importlib.resources.files` so the resolution
works both from a source checkout and an installed wheel (py3.10-safe).
"""

from __future__ import annotations

from importlib.resources import as_file, files
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

if TYPE_CHECKING:
    from jinja2 import Template

#: Name of the top-level template rendered into the final report.
TEMPLATE_NAME = "report.html.j2"


def _templates_dir() -> str:
    """Return the on-disk path of the bundled templates directory."""
    resource = files("dbsprout.report") / "templates"
    with as_file(resource) as path:
        return str(path)


def build_environment() -> Environment:
    """Construct the Jinja2 :class:`Environment` for report templates."""
    return Environment(
        loader=FileSystemLoader(_templates_dir()),
        autoescape=select_autoescape(["html", "htm", "xml", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def get_template() -> Template:
    """Load the top-level report template."""
    return build_environment().get_template(TEMPLATE_NAME)


def render_report(context: dict[str, Any]) -> str:
    """Render the report HTML for the given template ``context``."""
    return get_template().render(**context)
