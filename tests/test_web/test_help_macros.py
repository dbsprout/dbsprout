"""Shared tooltip / help macro tests (S-146).

The wizard and the Studio both grew per-field tooltips in S-123 (and a
field-descriptions JSON island for the Alpine method-picker). S-146
consolidates those one-off `title=` / `aria-describedby` snippets into a
single macros file `_help.html` so the same rendering path is reused on
both surfaces (and any future help-bearing template).

The macros are pure Jinja — no JS — so the tests render them through a
minimal :class:`Environment` rather than the full FastAPI app stack.
That keeps the unit suite fast and pins the macro contract independently
of the routes that consume it.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def env() -> object:
    """Return a Jinja env rooted at the web templates dir.

    Macros are loaded via ``{% import %}`` so the test stays close to how
    real templates consume them.
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: PLC0415

    templates_dir = Path(__file__).resolve().parents[2] / "dbsprout" / "web" / "templates"
    assert templates_dir.is_dir(), templates_dir
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html"]),
    )


def _render_tooltip(env: object, **kwargs: object) -> str:
    """Render the `tooltip` macro inline through a tiny wrapper template."""
    src = (
        "{% from '_help.html' import tooltip %}"
        "{{ tooltip(label=label, body=body, id_prefix=id_prefix) }}"
    )
    return env.from_string(src).render(**kwargs)  # type: ignore[attr-defined,no-any-return]


def _render_step_help(env: object, **kwargs: object) -> str:
    """Render the `step_help` macro through a tiny wrapper template."""
    src = (
        "{% from '_help.html' import step_help %}"
        "{{ step_help(step_n=step_n, step_label=step_label, blurb=blurb, generators=generators) }}"
    )
    return env.from_string(src).render(**kwargs)  # type: ignore[attr-defined,no-any-return]


# ── tooltip macro ────────────────────────────────────────────────────────


def test_tooltip_macro_emits_title_and_aria(env: object) -> None:
    """``tooltip`` carries both ``title=`` and ``aria-describedby``."""
    out = _render_tooltip(env, label="Email", body="A valid email.", id_prefix="t")
    assert "title=" in out
    assert "aria-describedby=" in out
    # The id used by aria-describedby is derived from the prefix + a slug
    # of the label so multiple tooltips on the same page don't collide.
    assert "id=" in out


def test_tooltip_macro_emits_sr_only_description(env: object) -> None:
    """The macro exposes the body inside an ``sr-only`` span for AT users."""
    out = _render_tooltip(env, label="Email", body="A valid email.", id_prefix="t")
    assert "sr-only" in out
    assert "A valid email." in out


def test_tooltip_macro_escapes_html_in_label(env: object) -> None:
    """User-supplied label / body strings must be HTML-escaped."""
    out = _render_tooltip(
        env,
        label="<script>bad()</script>",
        body="<img src=x onerror=alert(1)>",
        id_prefix="t",
    )
    assert "<script>bad" not in out
    assert "&lt;script&gt;bad" in out
    assert "<img " not in out
    assert "&lt;img" in out


def test_tooltip_macro_uses_id_prefix(env: object) -> None:
    """``id_prefix`` lets callers namespace ids inside a render loop."""
    a = _render_tooltip(env, label="A", body="b1", id_prefix="row-1")
    b = _render_tooltip(env, label="A", body="b2", id_prefix="row-2")
    assert "row-1" in a
    assert "row-2" in b
    # ids must differ — picker swaps a single row at a time so collisions
    # would break aria-describedby resolution.
    import re  # noqa: PLC0415

    id_a = re.search(r'id="([^"]+)"', a)
    id_b = re.search(r'id="([^"]+)"', b)
    assert id_a is not None
    assert id_b is not None
    assert id_a.group(1) != id_b.group(1)


# ── step_help macro ──────────────────────────────────────────────────────


def test_step_help_macro_emits_details_with_step_attribute(env: object) -> None:
    """``step_help`` renders a `<details>` collapsible with a stable hook."""
    out = _render_step_help(
        env,
        step_n=3,
        step_label="Configure",
        blurb="Configure how DBSprout generates data per column.",
        generators=[("mimesis", "email"), ("builtin", "random_int")],
    )
    # Stable selector so wizard tests can assert against it.
    assert 'data-help-step="3"' in out
    # The collapsible uses native <details>/<summary> for built-in
    # keyboard accessibility (focusable summary + Enter/Space toggle).
    assert "<details" in out
    assert "<summary" in out


def test_step_help_macro_renders_blurb_and_generators(env: object) -> None:
    """The popover surfaces the blurb + each generator name."""
    out = _render_step_help(
        env,
        step_n=3,
        step_label="Configure",
        blurb="Configure how DBSprout generates data per column.",
        generators=[("mimesis", "email"), ("builtin", "random_int")],
    )
    assert "Configure how DBSprout" in out
    assert "email" in out
    assert "random_int" in out


def test_step_help_macro_uses_question_mark_trigger(env: object) -> None:
    """The trigger is a ``?`` icon (text content) for visual + screen-reader cues."""
    out = _render_step_help(
        env,
        step_n=1,
        step_label="Connect",
        blurb="Pick a database or upload a schema file.",
        generators=[],
    )
    # ``?`` is the visible trigger glyph.
    assert "?" in out
    # The trigger also advertises an accessible name so screen readers
    # don't read a bare ``?``.
    assert "aria-label" in out


def test_step_help_macro_escapes_blurb(env: object) -> None:
    """The blurb is HTML-escaped (no raw HTML injection)."""
    out = _render_step_help(
        env,
        step_n=1,
        step_label="Connect",
        blurb="<script>x</script>",
        generators=[],
    )
    assert "<script>x</script>" not in out
    assert "&lt;script&gt;x&lt;/script&gt;" in out


def test_step_help_macro_handles_empty_generator_list(env: object) -> None:
    """An empty generator list renders without crashing."""
    out = _render_step_help(
        env,
        step_n=4,
        step_label="Generate",
        blurb="Click Run to generate rows.",
        generators=[],
    )
    assert 'data-help-step="4"' in out
    assert "Click Run" in out
