"""Per-column method-swap → instant re-preview JS behaviour (S-132).

The ``spec_grid_rerender.js`` module wires the S-120 method-picker swap to the
S-131 regenerate route and the S-147 preview route, debouncing rapid method
swaps and surfacing failures inline.

The module is driven under Node via ``require`` (same pattern as S-135's
``test_focus_cell_js.py`` and S-125's ``test_studio_console_js.py``) — no
jsdom, no Playwright. A tiny DOM stub, a fake ``window`` with a real event
bus, fake timers, and a stub ``fetch`` make the AC verifiable without a
browser. Each test ``console.log``\\s a JSON verdict and the Python side
asserts on the JSON.

The module exposes ``attachSpecGridRerender(opts?)``. Injectable hooks:

* ``_fetch`` — replaces ``window.fetch``.
* ``_setTimeout`` / ``_clearTimeout`` — drive the debounce window.
* ``_window`` — replaces ``window`` (event bus).
* ``_document`` — replaces ``document`` (DOM lookup for the affected row).
* ``debounceMs`` — overrides the 150 ms default.
* ``previewLimit`` — overrides the ``?limit=`` query param.

Events the module *listens* on (``window``):

* ``studio:row-rerendered`` — ``{table, column, method?, provider?}``
  Dispatched by the method-picker after it has swapped the spec row.

Events the module *dispatches* on ``window``:

* ``studio:preview-updated`` — ``{table, column, values, total}``
  ``values`` is the bounded slice for ONLY the column under change.
* ``studio:preview-error`` — ``{table, column, message}``
  Non-blocking. The spec is NOT reverted.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

if shutil.which("node") is None:  # pragma: no cover - environment guard
    pytest.skip(
        "node is not available; skipping JS-driven spec-grid rerender tests",
        allow_module_level=True,
    )


def _js_path() -> Path:
    from dbsprout.web import app as web_app  # noqa: PLC0415

    return Path(web_app.__file__).resolve().parent / "static" / "spec_grid_rerender.js"


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
    """Wrap *body* with a require + DOM/window/fetch/timer stubs.

    Exposed helpers inside the script:

    * ``mkElement(tag, attrs?, children?)`` — minimal element factory with
      ``setAttribute``/``getAttribute``/``hasAttribute``/``removeAttribute``
      so the module can write ``data-regenerating`` to a row.
    * ``mkDocument(rows)`` — builds a fake ``document`` whose
      ``querySelector('[data-column="<t>.<c>"]')`` returns the matching row.
    * ``windowEvents`` — array of every ``CustomEvent`` dispatched on
      ``win`` (so tests can assert on dispatched detail payloads).
    * ``win`` — fake window with ``addEventListener`` / ``dispatchEvent``.
    * ``advanceTime(ms)`` — drives the injected fake clock + timers.
    * ``fetchCalls`` — array of every ``{url, init}`` recorded by the stub.
    * ``mkFetch(responses)`` — build a fake fetch that consumes *responses*
      one per call. Each response is either ``{ok: true, json: ...}`` or
      ``{ok: false, status: ...}``.
    * ``attachSpecGridRerender`` — function under test.
    """
    js_path = _js_path()
    return f"""
        const path = {json.dumps(str(js_path))};

        // ── minimal DOM element stub ────────────────────────────────────
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
                _parent: null,
                setAttribute(name, value) {{
                    this._attrs[name] = String(value);
                    if (name.startsWith('data-')) {{
                        const key = name.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
                        this.dataset[key] = String(value);
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
                removeAttribute(name) {{
                    delete this._attrs[name];
                    if (name.startsWith('data-')) {{
                        const key = name.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
                        delete this.dataset[key];
                    }}
                }},
                appendChild(c) {{ c._parent = this; this.children.push(c); return c; }},
            }};
            const a = attrs || {{}};
            for (const k of Object.keys(a)) el.setAttribute(k, a[k]);
            const kids = children || [];
            for (const c of kids) el.appendChild(c);
            return el;
        }}

        // Tiny document stub: rows registered by their ``data-column``
        // attribute. ``querySelector`` matches the exact ``[data-column="X"]``
        // form the module uses.
        function mkDocument(rowMap) {{
            return {{
                querySelector(sel) {{
                    const m = sel.match(/^\\[data-column="(.+)"\\]$/);
                    if (!m) return null;
                    return Object.prototype.hasOwnProperty.call(rowMap, m[1])
                        ? rowMap[m[1]]
                        : null;
                }},
            }};
        }}

        // ── fake window + event bus ─────────────────────────────────────
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

        // ── injectable fake clock + timers ──────────────────────────────
        let _clock = 0;
        const _timers = [];
        let _nextTimerId = 1;
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
            due.sort((a, b) => a.due - b.due);
            for (const t of due) {{
                t.cancelled = true;
                t.cb();
            }}
        }}

        // ── fake fetch returning a queue of pre-baked responses ─────────
        // Each response is consumed in order; an empty queue yields a
        // failure (helps catch unexpected extra calls).
        const fetchCalls = [];
        function mkFetch(responses) {{
            const queue = (responses || []).slice();
            return function fakeFetch(url, init) {{
                fetchCalls.push({{ url: String(url), init: init || {{}} }});
                if (queue.length === 0) {{
                    return Promise.resolve({{
                        ok: false,
                        status: 599,
                        text: () => Promise.resolve('no stubbed response'),
                        json: () => Promise.resolve({{}}),
                    }});
                }}
                const next = queue.shift();
                if (next.reject) {{
                    return Promise.reject(new Error(next.reject));
                }}
                return Promise.resolve({{
                    ok: next.ok !== false,
                    status: next.status || 200,
                    text: () => Promise.resolve(next.text || ''),
                    json: () => Promise.resolve(next.json || {{}}),
                }});
            }};
        }}

        // Drain microtasks N times — Node has no "await all microtasks"
        // primitive, so we repeatedly ``await Promise.resolve()`` to flush.
        async function flushMicrotasks(n) {{
            for (let i = 0; i < (n || 10); i++) await Promise.resolve();
        }}

        const mod = require(path);
        const attachSpecGridRerender = mod.attachSpecGridRerender || win.attachSpecGridRerender;
        if (typeof attachSpecGridRerender !== 'function') {{
            throw new Error('attachSpecGridRerender not exported');
        }}

        function mkOpts(extra) {{
            const base = {{
                _setTimeout: fakeSetTimeout,
                _clearTimeout: fakeClearTimeout,
                _window: win,
                debounceMs: 150,
            }};
            const out = {{}};
            for (const k of Object.keys(base)) out[k] = base[k];
            for (const k of Object.keys(extra || {{}})) out[k] = extra[k];
            return out;
        }}

        (async () => {{
            {body}
        }})().catch((err) => {{
            console.log(JSON.stringify({{ _error: String(err && err.message || err) }}));
            process.exit(1);
        }});
    """


def test_module_exports_attach_function() -> None:
    """``attachSpecGridRerender`` is available on both ``module.exports`` and ``window``."""
    script = _harness("""
        console.log(JSON.stringify({
            modExport: typeof (require(path).attachSpecGridRerender),
            winExport: typeof win.attachSpecGridRerender,
        }));
    """)
    out = _run_node(script)
    assert out["modExport"] == "function"
    assert out["winExport"] == "function"


def test_row_rerendered_event_triggers_regen_post() -> None:
    """A single ``studio:row-rerendered`` event fires exactly one ``POST /api/regenerate``."""
    script = _harness("""
        const row = mkElement('div', {'data-column': 'users.email'});
        const doc = mkDocument({'users.email': row});
        const fetchFn = mkFetch([
            { ok: true, json: { kind: 'sync', table: 'users', column: 'email',
                                rows_affected: 100, rows: [{email: 'a@x'}] }},
            { ok: true, json: { table: 'users', limit: 100, total: 100,
                                rows: [{email: 'a@x'}, {email: 'b@x'}] }},
        ]);
        attachSpecGridRerender(mkOpts({ _fetch: fetchFn, _document: doc }));
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered', {
            detail: { table: 'users', column: 'email', method: 'email', provider: 'mimesis' },
        }));
        advanceTime(150);
        await flushMicrotasks(20);
        const posts = fetchCalls.filter(c => (c.init && c.init.method) === 'POST');
        const body = posts[0] ? JSON.parse(posts[0].init.body) : null;
        console.log(JSON.stringify({
            postCount: posts.length,
            url: posts[0] && posts[0].url,
            body,
        }));
    """)
    out = _run_node(script)
    assert out["postCount"] == 1
    assert out["url"] == "/api/regenerate"
    assert out["body"] == {"table": "users", "column": "email", "reroll": 1}


def test_debounce_coalesces_rapid_events_to_last() -> None:
    """Three rapid row-rerendered events on the same column → ONE regen call."""
    script = _harness("""
        const row = mkElement('div', {'data-column': 'users.email'});
        const doc = mkDocument({'users.email': row});
        const fetchFn = mkFetch([
            { ok: true, json: { kind: 'sync', rows: [] }},
            { ok: true, json: { table: 'users', limit: 100, total: 0, rows: [] }},
        ]);
        attachSpecGridRerender(mkOpts({ _fetch: fetchFn, _document: doc }));
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered',
            { detail: { table: 'users', column: 'email' } }));
        advanceTime(50);
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered',
            { detail: { table: 'users', column: 'email' } }));
        advanceTime(50);
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered',
            { detail: { table: 'users', column: 'email' } }));
        // Now wait past the full debounce window from the LAST event.
        advanceTime(200);
        await flushMicrotasks(20);
        const posts = fetchCalls.filter(c => (c.init && c.init.method) === 'POST');
        console.log(JSON.stringify({ postCount: posts.length }));
    """)
    out = _run_node(script)
    assert out["postCount"] == 1


def test_debounce_independent_per_column() -> None:
    """Two columns fired together both produce one regen call each (no cross-column coalescing)."""
    script = _harness("""
        const rowA = mkElement('div', {'data-column': 'users.email'});
        const rowB = mkElement('div', {'data-column': 'users.name'});
        const doc = mkDocument({'users.email': rowA, 'users.name': rowB});
        const fetchFn = mkFetch([
            { ok: true, json: { kind: 'sync', rows: [] }},
            { ok: true, json: { table: 'users', limit: 100, total: 0, rows: [] }},
            { ok: true, json: { kind: 'sync', rows: [] }},
            { ok: true, json: { table: 'users', limit: 100, total: 0, rows: [] }},
        ]);
        attachSpecGridRerender(mkOpts({ _fetch: fetchFn, _document: doc }));
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered',
            { detail: { table: 'users', column: 'email' } }));
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered',
            { detail: { table: 'users', column: 'name' } }));
        advanceTime(200);
        await flushMicrotasks(40);
        const posts = fetchCalls.filter(c => (c.init && c.init.method) === 'POST');
        const cols = posts.map(p => JSON.parse(p.init.body).column).sort();
        console.log(JSON.stringify({ postCount: posts.length, cols }));
    """)
    out = _run_node(script)
    assert out["postCount"] == 2
    assert out["cols"] == ["email", "name"]


def test_loading_state_set_on_row_and_cleared_on_success() -> None:
    """The affected row gets ``data-regenerating="true"`` during in-flight, cleared on done."""
    script = _harness("""
        const row = mkElement('div', {'data-column': 'users.email'});
        const doc = mkDocument({'users.email': row});
        // Track state via a deferred promise so we can observe mid-flight.
        let resolveRegen;
        const regenPromise = new Promise((res) => { resolveRegen = res; });
        const fetchFn = (url, init) => {
            fetchCalls.push({ url: String(url), init: init || {} });
            if (String(url) === '/api/regenerate') {
                return regenPromise.then(() => ({
                    ok: true, status: 200,
                    text: () => Promise.resolve(''),
                    json: () => Promise.resolve({ kind: 'sync', rows: [] }),
                }));
            }
            return Promise.resolve({
                ok: true, status: 200,
                text: () => Promise.resolve(''),
                json: () => Promise.resolve({ table: 'users', limit: 100, total: 0, rows: [] }),
            });
        };
        attachSpecGridRerender(mkOpts({ _fetch: fetchFn, _document: doc }));
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered',
            { detail: { table: 'users', column: 'email' } }));
        advanceTime(150);
        await flushMicrotasks(10);
        const midFlight = row.getAttribute('data-regenerating');
        resolveRegen();
        await flushMicrotasks(20);
        const afterDone = row.getAttribute('data-regenerating');
        console.log(JSON.stringify({ midFlight, afterDone }));
    """)
    out = _run_node(script)
    assert out["midFlight"] == "true"
    assert out["afterDone"] is None


def test_loading_state_cleared_on_failure() -> None:
    """The loading attribute is also cleared when regen fails (non-blocking error)."""
    script = _harness("""
        const row = mkElement('div', {'data-column': 'users.email'});
        const doc = mkDocument({'users.email': row});
        const fetchFn = mkFetch([
            { ok: false, status: 500, text: 'boom' },  // regen fails
        ]);
        attachSpecGridRerender(mkOpts({ _fetch: fetchFn, _document: doc }));
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered',
            { detail: { table: 'users', column: 'email' } }));
        advanceTime(150);
        await flushMicrotasks(20);
        const after = row.getAttribute('data-regenerating');
        console.log(JSON.stringify({ after }));
    """)
    out = _run_node(script)
    assert out["after"] is None


def test_preview_fetch_follows_regen_success() -> None:
    """After regen ok, ``GET /api/preview/{table}?limit=N`` is called."""
    script = _harness("""
        const row = mkElement('div', {'data-column': 'users.email'});
        const doc = mkDocument({'users.email': row});
        const fetchFn = mkFetch([
            { ok: true, json: { kind: 'sync', rows: [] }},
            { ok: true, json: { table: 'users', limit: 100, total: 0, rows: [] }},
        ]);
        attachSpecGridRerender(mkOpts({ _fetch: fetchFn, _document: doc, previewLimit: 100 }));
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered',
            { detail: { table: 'users', column: 'email' } }));
        advanceTime(150);
        await flushMicrotasks(20);
        const gets = fetchCalls.filter(c => !(c.init && c.init.method) || c.init.method === 'GET');
        console.log(JSON.stringify({
            getCount: gets.length,
            url: gets[0] && gets[0].url,
        }));
    """)
    out = _run_node(script)
    assert out["getCount"] == 1
    assert out["url"] == "/api/preview/users?limit=100"


def test_preview_updated_event_includes_column_slice_only() -> None:
    """The dispatched ``studio:preview-updated`` carries values for ONLY the swapped column."""
    script = _harness("""
        const row = mkElement('div', {'data-column': 'users.email'});
        const doc = mkDocument({'users.email': row});
        const fetchFn = mkFetch([
            { ok: true, json: { kind: 'sync', rows: [] }},
            { ok: true, json: {
                table: 'users', limit: 100, total: 2,
                rows: [
                    { id: 1, email: 'a@x.com', name: 'A' },
                    { id: 2, email: 'b@x.com', name: 'B' },
                ],
            }},
        ]);
        attachSpecGridRerender(mkOpts({ _fetch: fetchFn, _document: doc }));
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered',
            { detail: { table: 'users', column: 'email' } }));
        advanceTime(150);
        await flushMicrotasks(30);
        const evt = windowEvents.find(e => e.type === 'studio:preview-updated');
        console.log(JSON.stringify({ detail: evt && evt.detail }));
    """)
    out = _run_node(script)
    assert isinstance(out["detail"], dict)
    detail = out["detail"]
    assert detail["table"] == "users"
    assert detail["column"] == "email"
    assert detail["values"] == ["a@x.com", "b@x.com"]
    assert detail["total"] == 2


def test_regen_failure_emits_preview_error_and_skips_preview_fetch() -> None:
    """Regen failure → ``studio:preview-error`` and no preview GET."""
    script = _harness("""
        const row = mkElement('div', {'data-column': 'users.email'});
        const doc = mkDocument({'users.email': row});
        const fetchFn = mkFetch([
            { ok: false, status: 409, text: 'constraint' },
        ]);
        attachSpecGridRerender(mkOpts({ _fetch: fetchFn, _document: doc }));
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered',
            { detail: { table: 'users', column: 'email' } }));
        advanceTime(150);
        await flushMicrotasks(20);
        const errEvt = windowEvents.find(e => e.type === 'studio:preview-error');
        const updEvt = windowEvents.find(e => e.type === 'studio:preview-updated');
        const gets = fetchCalls.filter(c => !(c.init && c.init.method) || c.init.method === 'GET');
        console.log(JSON.stringify({
            errDetail: errEvt && errEvt.detail,
            updPresent: !!updEvt,
            getCount: gets.length,
        }));
    """)
    out = _run_node(script)
    assert isinstance(out["errDetail"], dict)
    assert out["errDetail"]["table"] == "users"
    assert out["errDetail"]["column"] == "email"
    assert "message" in out["errDetail"]
    assert out["updPresent"] is False
    assert out["getCount"] == 0


def test_preview_failure_emits_preview_error_after_successful_regen() -> None:
    """Preview GET failure (after regen ok) → error event, still no spec revert."""
    script = _harness("""
        const row = mkElement('div', {'data-column': 'users.email'});
        const doc = mkDocument({'users.email': row});
        const fetchFn = mkFetch([
            { ok: true, json: { kind: 'sync', rows: [] }},
            { ok: false, status: 500, text: 'boom' },
        ]);
        attachSpecGridRerender(mkOpts({ _fetch: fetchFn, _document: doc }));
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered',
            { detail: { table: 'users', column: 'email' } }));
        advanceTime(150);
        await flushMicrotasks(20);
        const errEvt = windowEvents.find(e => e.type === 'studio:preview-error');
        const updEvt = windowEvents.find(e => e.type === 'studio:preview-updated');
        console.log(JSON.stringify({
            errPresent: !!errEvt,
            errCol: errEvt && errEvt.detail && errEvt.detail.column,
            updPresent: !!updEvt,
        }));
    """)
    out = _run_node(script)
    assert out["errPresent"] is True
    assert out["errCol"] == "email"
    assert out["updPresent"] is False


def test_untouched_row_invariant_during_swap() -> None:
    """A sibling row in the same table is NOT mutated by another column's regen."""
    script = _harness("""
        const target = mkElement('div', {'data-column': 'users.email'});
        const sibling = mkElement('div', {'data-column': 'users.name'});
        const doc = mkDocument({'users.email': target, 'users.name': sibling});
        let resolveRegen;
        const regenPromise = new Promise((res) => { resolveRegen = res; });
        const fetchFn = (url, init) => {
            fetchCalls.push({ url: String(url), init: init || {} });
            if (String(url) === '/api/regenerate') {
                return regenPromise.then(() => ({
                    ok: true, status: 200,
                    text: () => Promise.resolve(''),
                    json: () => Promise.resolve({ kind: 'sync', rows: [] }),
                }));
            }
            return Promise.resolve({
                ok: true, status: 200,
                text: () => Promise.resolve(''),
                json: () => Promise.resolve({ table: 'users', limit: 100, total: 0, rows: [] }),
            });
        };
        attachSpecGridRerender(mkOpts({ _fetch: fetchFn, _document: doc }));
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered',
            { detail: { table: 'users', column: 'email' } }));
        advanceTime(150);
        await flushMicrotasks(10);
        const siblingMid = sibling.getAttribute('data-regenerating');
        const targetMid = target.getAttribute('data-regenerating');
        resolveRegen();
        await flushMicrotasks(20);
        console.log(JSON.stringify({
            siblingMid, targetMid,
            siblingAttrs: Object.keys(sibling._attrs),
        }));
    """)
    out = _run_node(script)
    assert out["siblingMid"] is None
    assert out["targetMid"] == "true"
    # The sibling row's attributes were left exactly as we wrote them
    # (only the original ``data-column``).
    assert out["siblingAttrs"] == ["data-column"]


def test_nonce_bumps_each_non_coalesced_call() -> None:
    """Two successive non-coalesced regen calls send ``reroll: 1`` then ``reroll: 2``."""
    script = _harness("""
        const row = mkElement('div', {'data-column': 'users.email'});
        const doc = mkDocument({'users.email': row});
        const fetchFn = mkFetch([
            { ok: true, json: { kind: 'sync', rows: [] }},
            { ok: true, json: { table: 'users', limit: 100, total: 0, rows: [] }},
            { ok: true, json: { kind: 'sync', rows: [] }},
            { ok: true, json: { table: 'users', limit: 100, total: 0, rows: [] }},
        ]);
        attachSpecGridRerender(mkOpts({ _fetch: fetchFn, _document: doc }));
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered',
            { detail: { table: 'users', column: 'email' } }));
        advanceTime(200);
        await flushMicrotasks(20);
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered',
            { detail: { table: 'users', column: 'email' } }));
        advanceTime(200);
        await flushMicrotasks(20);
        const posts = fetchCalls.filter(c => (c.init && c.init.method) === 'POST');
        const rerolls = posts.map(p => JSON.parse(p.init.body).reroll);
        console.log(JSON.stringify({ postCount: posts.length, rerolls }));
    """)
    out = _run_node(script)
    assert out["postCount"] == 2
    assert out["rerolls"] == [1, 2]


def test_event_with_missing_table_or_column_is_noop() -> None:
    """Defensive: malformed event detail → no fetch, no DOM mutation."""
    script = _harness("""
        const doc = mkDocument({});
        const fetchFn = mkFetch([]);
        attachSpecGridRerender(mkOpts({ _fetch: fetchFn, _document: doc }));
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered',
            { detail: { table: '', column: 'email' } }));
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered',
            { detail: { table: 'users', column: '' } }));
        win.dispatchEvent(new win.CustomEvent('studio:row-rerendered',
            { detail: null }));
        advanceTime(300);
        await flushMicrotasks(20);
        console.log(JSON.stringify({ calls: fetchCalls.length }));
    """)
    out = _run_node(script)
    assert out["calls"] == 0
