"""Violation drill-down JS behaviour — driven under Node (S-135).

The focus-cell module lives in ``dbsprout/web/static/focus_cell.js``. To prove
the client-side AC (event dispatch payload, keyboard accessibility, listener
DOM mutations, flash class lifecycle) without booting a browser we spawn
``node`` and ``require()`` the file. The module is designed to:

* expose ``window.attachFocusCell`` / ``window.installFocusCell`` and also
  return ``{attachFocusCell, installFocusCell}`` from ``module.exports`` when
  ``module`` is present (so the file is both a browser global and a CommonJS
  module under Node).
* accept an injectable ``_now`` / ``_setTimeout`` / ``_clearTimeout`` hook
  bag so the test can drive the flash-class timing deterministically.

A single Node script exercises each JS-side AC and ``console.log``\\s a JSON
verdict; the Python test asserts on that JSON. This keeps the JS test surface
tiny (no jsdom, no Playwright) and runs in well under a second per test.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

if shutil.which("node") is None:  # pragma: no cover - environment guard
    pytest.skip(
        "node is not available; skipping JS-driven focus-cell tests",
        allow_module_level=True,
    )


def _js_path() -> Path:
    from dbsprout.web import app as web_app  # noqa: PLC0415

    return Path(web_app.__file__).resolve().parent / "static" / "focus_cell.js"


def _run_node(script: str) -> dict[str, object]:
    """Run *script* under Node, expecting one JSON object on stdout (last line)."""
    result = subprocess.run(  # noqa: S603 - fixed argv, trusted interpreter
        ["node", "--input-type=commonjs", "-e", script],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        f"node script failed (exit {result.returncode}): stderr={result.stderr}"
    )
    last = result.stdout.strip().splitlines()[-1]
    return json.loads(last)


def _harness(body: str) -> str:
    """Wrap *body* with a require + minimal DOM harness.

    The harness builds a tiny dependency-free DOM stub (only the surface
    the focus-cell module touches) so we don't need jsdom. Exposed helpers:

      * ``attachFocusCell`` / ``installFocusCell`` — the functions under test.
      * ``mkElement(tag, attrs?, children?)`` — minimal element factory.
      * ``windowEvents`` — array capturing every ``CustomEvent`` the module
        dispatches on ``window`` (so tests can assert on detail payloads).
      * ``advanceTime(ms)`` — drives the injected fake clock + timers.
    """
    js_path = _js_path()
    return f"""
        const path = {json.dumps(str(js_path))};

        // ── minimal DOM stub (only what focus_cell.js touches) ──────────
        let _nextId = 1;
        function mkElement(tag, attrs, children) {{
            const el = {{
                tagName: tag.toUpperCase(),
                _id: _nextId++,
                _attrs: {{}},
                dataset: {{}},
                classList: {{
                    _set: new Set(),
                    add(c) {{ this._set.add(c); }},
                    remove(c) {{ this._set.delete(c); }},
                    contains(c) {{ return this._set.has(c); }},
                }},
                children: [],
                _listeners: {{}},
                _parent: null,
                _scrollCount: 0,
                _focusCalls: 0,
                scrollIntoView() {{ this._scrollCount++; }},
                focus() {{ this._focusCalls++; }},
                addEventListener(name, fn) {{
                    (this._listeners[name] = this._listeners[name] || []).push(fn);
                }},
                removeEventListener(name, fn) {{
                    const arr = this._listeners[name] || [];
                    const i = arr.indexOf(fn);
                    if (i >= 0) arr.splice(i, 1);
                }},
                dispatchEvent(ev) {{
                    const arr = this._listeners[ev.type] || [];
                    for (const fn of arr.slice()) fn(ev);
                    return true;
                }},
                setAttribute(name, value) {{
                    this._attrs[name] = String(value);
                    if (name.startsWith('data-')) {{
                        const key = name.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
                        this.dataset[key] = String(value);
                    }}
                    if (name === 'class') {{
                        this.classList._set = new Set(
                            String(value).split(/\\s+/).filter(Boolean)
                        );
                    }}
                }},
                getAttribute(name) {{
                    return Object.prototype.hasOwnProperty.call(this._attrs, name)
                        ? this._attrs[name]
                        : null;
                }},
                hasAttribute(name) {{
                    return Object.prototype.hasOwnProperty.call(this._attrs, name);
                }},
                removeAttribute(name) {{ delete this._attrs[name]; }},
                appendChild(c) {{ c._parent = this; this.children.push(c); return c; }},
                querySelectorAll(sel) {{ return _querySelectorAll(this, sel); }},
                querySelector(sel) {{
                    const all = _querySelectorAll(this, sel);
                    return all.length ? all[0] : null;
                }},
            }};
            const a = attrs || {{}};
            for (const k of Object.keys(a)) el.setAttribute(k, a[k]);
            const kids = children || [];
            for (const c of kids) el.appendChild(c);
            return el;
        }}

        // Walk the tree and apply a tiny selector grammar:
        //   * "[data-foo]"
        //   * "[data-foo=\\"bar\\"]"
        //   * ".class"
        //   * a chain of those separated by spaces (descendant match — any
        //     descendant satisfying the chained predicates).
        function _matches(el, predicate) {{
            const attrEq = predicate.match(/^\\[([a-zA-Z-]+)="(.*)"\\]$/);
            if (attrEq) return el.getAttribute(attrEq[1]) === attrEq[2];
            const attr = predicate.match(/^\\[([a-zA-Z-]+)\\]$/);
            if (attr) return el.hasAttribute(attr[1]);
            const cls = predicate.match(/^\\.(.+)$/);
            if (cls) return el.classList.contains(cls[1]);
            return false;
        }}
        function _walk(root, visit) {{
            visit(root);
            for (const c of root.children) _walk(c, visit);
        }}
        function _querySelectorAll(root, sel) {{
            // Compound (space-separated): an element matches the LAST predicate AND
            // has some ancestor matching each previous predicate in order.
            const parts = sel.trim().split(/\\s+/);
            const last = parts[parts.length - 1];
            const ancestorPreds = parts.slice(0, -1);
            const matches = [];
            _walk(root, (el) => {{
                if (el === root) return;
                if (!_matches(el, last)) return;
                if (ancestorPreds.length === 0) {{ matches.push(el); return; }}
                let p = el._parent;
                let i = ancestorPreds.length - 1;
                while (p && i >= 0) {{
                    if (_matches(p, ancestorPreds[i])) i--;
                    p = p._parent;
                }}
                if (i < 0) matches.push(el);
            }});
            return matches;
        }}

        // ── minimal window with event dispatch ──────────────────────────
        const windowEvents = [];
        const windowListeners = {{}};
        const win = {{
            addEventListener(name, fn) {{
                (windowListeners[name] = windowListeners[name] || []).push(fn);
            }},
            removeEventListener(name, fn) {{
                const arr = windowListeners[name] || [];
                const i = arr.indexOf(fn);
                if (i >= 0) arr.splice(i, 1);
            }},
            dispatchEvent(ev) {{
                windowEvents.push(ev);
                const arr = windowListeners[ev.type] || [];
                for (const fn of arr.slice()) fn(ev);
                return true;
            }},
            CustomEvent: function (type, init) {{
                return {{ type, detail: (init && init.detail) || null }};
            }},
        }};
        global.window = win;
        global.CustomEvent = win.CustomEvent;

        // ── injectable clock + timers ──────────────────────────────────
        let _clock = 0;
        const _timers = []; // {{id, due, cb, cancelled}}
        let _nextTimerId = 1;
        const fakeNow = () => _clock;
        const fakeSetTimeout = (cb, ms) => {{
            const t = {{ id: _nextTimerId++, due: _clock + ms, cb, cancelled: false }};
            _timers.push(t);
            return t.id;
        }};
        const fakeClearTimeout = (id) => {{
            for (const t of _timers) if (t.id === id) t.cancelled = true;
        }};
        function advanceTime(ms) {{
            _clock += ms;
            const due = _timers.filter(t => !t.cancelled && t.due <= _clock);
            // Fire in due order; we mark them cancelled to avoid re-firing.
            due.sort((a, b) => a.due - b.due);
            for (const t of due) {{
                t.cancelled = true;
                t.cb();
            }}
        }}

        const mod = require(path);
        const attachFocusCell = mod.attachFocusCell || win.attachFocusCell;
        const installFocusCell = mod.installFocusCell || win.installFocusCell;
        if (typeof attachFocusCell !== 'function') {{
            throw new Error('attachFocusCell not exported');
        }}
        if (typeof installFocusCell !== 'function') {{
            throw new Error('installFocusCell not exported');
        }}

        const _hooks = {{
            _now: fakeNow,
            _setTimeout: fakeSetTimeout,
            _clearTimeout: fakeClearTimeout,
            _window: win,
        }};

        {body}
    """


def test_click_on_violation_row_dispatches_focus_cell_event() -> None:
    """Clicking a violation row → exactly one ``studio:focus-cell`` event with the right detail."""
    script = _harness("""
        const row = mkElement('tr', {
            class: 'violation-row',
            'data-table': 'orders',
            'data-column': 'user_id',
            'data-row': '7',
        });
        const panel = mkElement('div', {id: 'validate-panel'}, [row]);
        attachFocusCell(panel, _hooks);
        // Simulate click.
        row.dispatchEvent({type: 'click', currentTarget: row, target: row});
        const focusEvents = windowEvents.filter(e => e.type === 'studio:focus-cell');
        console.log(JSON.stringify({
            count: focusEvents.length,
            detail: focusEvents[0] && focusEvents[0].detail,
        }));
    """)
    out = _run_node(script)
    assert out["count"] == 1
    assert out["detail"] == {"table": "orders", "column": "user_id", "row": 7}


def test_empty_dataset_row_is_undefined_not_empty_string() -> None:
    """A violation row with ``data-row=""`` → event ``detail.row`` is missing/undefined."""
    script = _harness("""
        const row = mkElement('tr', {
            class: 'violation-row',
            'data-table': 'orders',
            'data-column': 'user_id',
            'data-row': '',
        });
        const panel = mkElement('div', {id: 'validate-panel'}, [row]);
        attachFocusCell(panel, _hooks);
        row.dispatchEvent({type: 'click', currentTarget: row, target: row});
        const ev = windowEvents.find(e => e.type === 'studio:focus-cell');
        // JSON.stringify drops `undefined` → omit-on-serialise == ok.
        console.log(JSON.stringify({
            detail: ev && ev.detail,
            hasRow: ev && Object.prototype.hasOwnProperty.call(ev.detail, 'row'),
        }));
    """)
    out = _run_node(script)
    assert out["detail"] == {"table": "orders", "column": "user_id"}
    assert out["hasRow"] is False


def test_enter_and_space_keypresses_dispatch_same_event() -> None:
    """``keydown`` with ``Enter`` or `` `` dispatches the same event; other keys do not."""
    script = _harness("""
        const row = mkElement('tr', {
            class: 'violation-row',
            'data-table': 't', 'data-column': 'c', 'data-row': '',
        });
        const panel = mkElement('div', {id: 'validate-panel'}, [row]);
        attachFocusCell(panel, _hooks);
        const prevented = [];
        const mkKey = (key) => ({
            type: 'keydown', key, currentTarget: row, target: row,
            preventDefault() { prevented.push(key); },
        });
        row.dispatchEvent(mkKey('Enter'));
        row.dispatchEvent(mkKey(' '));
        row.dispatchEvent(mkKey('a'));   // ignored
        row.dispatchEvent(mkKey('Tab')); // ignored
        const focusEvents = windowEvents.filter(e => e.type === 'studio:focus-cell');
        console.log(JSON.stringify({
            count: focusEvents.length,
            prevented,
        }));
    """)
    out = _run_node(script)
    assert out["count"] == 2
    assert out["prevented"] == ["Enter", " "]


def test_spec_grid_listener_marks_row_focused_and_adds_flash_class() -> None:
    """``studio:focus-cell`` → matching spec-grid row gains ``data-focused`` + flash class."""
    script = _harness("""
        const row = mkElement('div', {'data-column': 'orders.user_id'});
        const section = mkElement('section', {'data-table': 'orders'}, [row]);
        const grid = mkElement('div', {id: 'spec-grid'}, [section]);
        installFocusCell(grid, _hooks);
        win.dispatchEvent(new win.CustomEvent('studio:focus-cell', {
            detail: { table: 'orders', column: 'user_id' },
        }));
        console.log(JSON.stringify({
            focused: row.getAttribute('data-focused'),
            flash: row.classList.contains('studio-focus-flash'),
            scrolled: row._scrollCount,
        }));
    """)
    out = _run_node(script)
    assert out["focused"] == "true"
    assert out["flash"] is True
    assert out["scrolled"] == 1


def test_flash_class_removed_after_timeout_but_focus_persists() -> None:
    """After ~3 s the flash class clears but ``data-focused`` stays for next-event behaviour."""
    script = _harness("""
        const row = mkElement('div', {'data-column': 'orders.user_id'});
        const section = mkElement('section', {'data-table': 'orders'}, [row]);
        const grid = mkElement('div', {id: 'spec-grid'}, [section]);
        installFocusCell(grid, _hooks);
        win.dispatchEvent(new win.CustomEvent('studio:focus-cell', {
            detail: { table: 'orders', column: 'user_id' },
        }));
        // Flash on immediately.
        const before = row.classList.contains('studio-focus-flash');
        // Past the documented 3 s window.
        advanceTime(3100);
        const after = row.classList.contains('studio-focus-flash');
        console.log(JSON.stringify({
            before, after,
            focusedAfter: row.getAttribute('data-focused'),
        }));
    """)
    out = _run_node(script)
    assert out["before"] is True
    assert out["after"] is False
    assert out["focusedAfter"] == "true"


def test_dispatching_for_unknown_table_is_safe_noop() -> None:
    """An event for a table that isn't in the grid: no DOM mutation, no error."""
    script = _harness("""
        const row = mkElement('div', {'data-column': 'orders.user_id'});
        const section = mkElement('section', {'data-table': 'orders'}, [row]);
        const grid = mkElement('div', {id: 'spec-grid'}, [section]);
        installFocusCell(grid, _hooks);
        win.dispatchEvent(new win.CustomEvent('studio:focus-cell', {
            detail: { table: 'missing', column: 'user_id' },
        }));
        console.log(JSON.stringify({
            focused: row.getAttribute('data-focused'),
            flash: row.classList.contains('studio-focus-flash'),
            scrolled: row._scrollCount,
        }));
    """)
    out = _run_node(script)
    assert out["focused"] is None
    assert out["flash"] is False
    assert out["scrolled"] == 0


def test_preview_grid_row_scrolls_when_row_provided() -> None:
    """When ``row`` is in the event, the preview row's ``scrollIntoView`` fires."""
    script = _harness("""
        // Spec grid for parity (no match needed here).
        const grid = mkElement('div', {id: 'spec-grid'});
        installFocusCell(grid, _hooks);
        // Preview grid stub: rows tagged with table + row index.
        const target = mkElement('tr', {'data-row-index': '3'});
        const other = mkElement('tr', {'data-row-index': '5'});
        const preview = mkElement(
            'div',
            {id: 'preview', 'data-preview-table': 'orders'},
            [other, target]
        );
        installFocusCell(preview, _hooks);
        win.dispatchEvent(new win.CustomEvent('studio:focus-cell', {
            detail: { table: 'orders', column: 'user_id', row: 3 },
        }));
        console.log(JSON.stringify({
            targetScrolled: target._scrollCount,
            otherScrolled: other._scrollCount,
        }));
    """)
    out = _run_node(script)
    assert out["targetScrolled"] == 1
    assert out["otherScrolled"] == 0


def test_reinstall_listener_does_not_double_fire() -> None:
    """Calling ``installFocusCell`` twice on the same root: the listener fires once per event."""
    script = _harness("""
        const row = mkElement('div', {'data-column': 'orders.user_id'});
        const section = mkElement('section', {'data-table': 'orders'}, [row]);
        const grid = mkElement('div', {id: 'spec-grid'}, [section]);
        installFocusCell(grid, _hooks);
        installFocusCell(grid, _hooks); // re-install (HTMX afterSwap simulation)
        win.dispatchEvent(new win.CustomEvent('studio:focus-cell', {
            detail: { table: 'orders', column: 'user_id' },
        }));
        console.log(JSON.stringify({
            scrolled: row._scrollCount,
        }));
    """)
    out = _run_node(script)
    # Re-install must replace, not stack.
    assert out["scrolled"] == 1


def test_attach_marks_violation_rows_keyboard_accessible() -> None:
    """``attachFocusCell`` adds ``tabindex="0"`` and ``role="button"`` to violation rows."""
    script = _harness("""
        const row = mkElement('tr', {
            class: 'violation-row',
            'data-table': 't', 'data-column': 'c', 'data-row': '',
        });
        const panel = mkElement('div', {id: 'validate-panel'}, [row]);
        attachFocusCell(panel, _hooks);
        console.log(JSON.stringify({
            tabindex: row.getAttribute('tabindex'),
            role: row.getAttribute('role'),
        }));
    """)
    out = _run_node(script)
    assert out["tabindex"] == "0"
    assert out["role"] == "button"
