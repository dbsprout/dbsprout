"""Studio console JS behaviour — driven directly under Node (S-125).

The Alpine factory lives in ``dbsprout/web/static/studio_console.js``. To prove
the **client-side** AC (state transitions, rolling rows/sec, terminal freeze,
reconnect-once, fallback to GET /api/jobs/{id}) without booting a browser we
spawn ``node`` and ``require()`` the file. The factory is designed to:

* expose ``window.studioConsole`` and also return its callable from
  ``module.exports`` when ``module`` is present (so the file is both an Alpine
  factory in a browser and a CommonJS module under Node).
* accept an injectable ``wsFactory(url)`` and ``fetchFn(url)`` so the test can
  drive ``onmessage`` / ``onclose`` without a real socket.

A single Node script exercises every JS-side AC and ``console.log``\\s a JSON
verdict; the Python test asserts on that JSON. This keeps the JS test surface
tiny (no jsdom, no Playwright) and runs in well under a second.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

if shutil.which("node") is None:  # pragma: no cover - environment guard
    pytest.skip("node is not available; skipping JS-driven console tests", allow_module_level=True)


def _js_path() -> Path:
    from dbsprout.web import app as web_app  # noqa: PLC0415

    return Path(web_app.__file__).resolve().parent / "static" / "studio_console.js"


def _run_node(script: str) -> dict[str, object]:
    """Run *script* under Node, expecting one JSON object on stdout."""
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
    # The script prints a JSON verdict as its *last* line.
    last = result.stdout.strip().splitlines()[-1]
    return json.loads(last)


def _harness(body: str) -> str:
    """Wrap *body* with the require + fake-socket harness.

    Exposes:
      * ``studioConsole`` — the factory under test
      * ``FakeWS`` — captures the latest instance for assertion / driving
      * ``mkComponent({fetchFn})`` — instantiates the factory with injected hooks
    """
    js_path = _js_path()
    return f"""
        const path = {json.dumps(str(js_path))};
        // The factory file is written so that under Node it attaches the
        // ``studioConsole`` function to module.exports (and also to a
        // module-local ``global.window``) so we can require() it cleanly.
        global.window = global.window || {{}};
        const mod = require(path);
        const studioConsole = mod.studioConsole || global.window.studioConsole;
        if (typeof studioConsole !== 'function') {{
            throw new Error('studioConsole factory not exported');
        }}

        class FakeWS {{
            constructor(url) {{
                this.url = url;
                this.readyState = 1;
                FakeWS.last = this;
                FakeWS.instances.push(this);
                this.onopen = null;
                this.onmessage = null;
                this.onclose = null;
                this.onerror = null;
                this.sentClose = false;
            }}
            send() {{ /* unused */ }}
            close() {{ this.sentClose = true; this.readyState = 3; }}
            // helpers driven by the test:
            push(frame) {{ this.onmessage({{ data: JSON.stringify(frame) }}); }}
            drop(code = 1006) {{ this.readyState = 3; if (this.onclose) this.onclose({{ code }}); }}
        }}
        FakeWS.instances = [];
        FakeWS.last = null;

        function mkComponent(opts = {{}}) {{
            const c = studioConsole();
            c._wsFactory = (url) => new FakeWS(url);
            c._fetchFn = opts.fetchFn || (async () => ({{ ok: true, json: async () => ({{}}) }}));
            // Avoid relying on a real ``window`` for tests of the bare factory.
            c._now = opts.now || (() => Date.now());
            return c;
        }}

        {body}
    """


def test_table_start_initialises_table_row() -> None:
    """``table_start`` creates a per-table entry with zero rows + sensible defaults."""
    script = _harness("""
        const c = mkComponent();
        c.connect('job-1');
        FakeWS.last.push({phase: 'table_start', table: 'users', tables_done: 0, tables_total: 2});
        const t = c.tables.find(t => t.name === 'users');
        const out = {
            hasTable: !!t,
            rowsDone: t && t.rows_done,
            tablesTotal: c.tablesTotal,
            status: c.status,
        };
        console.log(JSON.stringify(out));
    """)
    out = _run_node(script)
    assert out == {
        "hasTable": True,
        "rowsDone": 0,
        "tablesTotal": 2,
        "status": "running",
    }


def test_table_done_advances_overall_and_rows_per_sec() -> None:
    """Two ``table_done`` frames yield correct overall % and a non-zero rolling rate."""
    script = _harness("""
        let t = 1000;
        const c = mkComponent({now: () => t});
        c.connect('job-2');
        FakeWS.last.push({phase: 'table_start', tables_total: 2});
        t += 1000;  // 1s elapsed
        FakeWS.last.push({
            phase: 'table_done', table: 'users',
            tables_done: 1, tables_total: 2,
            rows_in_table: 100, total_rows: 100,
        });
        t += 1000;  // 2s total
        FakeWS.last.push({
            phase: 'table_done', table: 'orders',
            tables_done: 2, tables_total: 2,
            rows_in_table: 100, total_rows: 200,
        });
        const out = {
            overall: c.overall,
            rowsPerSec: c.rowsPerSec,
            tableCount: c.tables.length,
            firstName: c.tables[0].name,
            secondName: c.tables[1].name,
        };
        console.log(JSON.stringify(out));
    """)
    out = _run_node(script)
    assert out["overall"] == 100  # 2/2 tables done
    assert out["tableCount"] == 2
    assert out["firstName"] == "users"
    assert out["secondName"] == "orders"
    assert isinstance(out["rowsPerSec"], (int, float))
    assert out["rowsPerSec"] > 0


def test_rolling_window_drops_older_than_five_samples() -> None:
    """Rolling mean uses the most recent 5 samples (oldest dropped)."""
    script = _harness("""
        let t = 0;
        const c = mkComponent({now: () => t});
        c.connect('job-rolling');
        // Seven samples; first two should fall out of the rolling window.
        for (let i = 1; i <= 7; i++) {
            t = i * 1000; // 1s apart
            FakeWS.last.push({
                phase: 'table_done', table: 't' + i,
                tables_done: i, tables_total: 7,
                rows_in_table: 1000, total_rows: 1000 * i,
            });
        }
        console.log(JSON.stringify({ samples: c._samples.length, rate: c.rowsPerSec }));
    """)
    out = _run_node(script)
    assert out["samples"] == 5, "rolling window must retain at most 5 samples"
    # All deltas are 1000 rows / 1s → rate is exactly 1000/s.
    assert out["rate"] == 1000


def test_terminal_freezes_status_and_bars() -> None:
    """A ``terminal`` frame freezes bars and sets the terminal status + summary."""
    script = _harness("""
        const c = mkComponent();
        c.connect('job-3');
        FakeWS.last.push({phase: 'table_start', table: 'a', tables_total: 1});
        FakeWS.last.push({
            phase: 'table_done', table: 'a',
            tables_done: 1, tables_total: 1,
            rows_in_table: 50, total_rows: 50,
        });
        FakeWS.last.push({phase: 'terminal', status: 'succeeded', error: null});
        const out = {
            status: c.status,
            frozen: c.frozen,
            overall: c.overall,
            summary: c.summary,
        };
        console.log(JSON.stringify(out));
    """)
    out = _run_node(script)
    assert out == {
        "status": "succeeded",
        "frozen": True,
        "overall": 100,
        "summary": "Succeeded · 1/1 tables · 50 rows",
    }


def test_terminal_failed_surfaces_error_in_summary() -> None:
    script = _harness("""
        const c = mkComponent();
        c.connect('job-fail');
        FakeWS.last.push({phase: 'table_start', table: 'x', tables_total: 1});
        FakeWS.last.push({phase: 'terminal', status: 'failed', error: 'boom'});
        console.log(JSON.stringify({status: c.status, summary: c.summary}));
    """)
    out = _run_node(script)
    assert out["status"] == "failed"
    assert "boom" in out["summary"]


def test_mid_stream_disconnect_triggers_one_reconnect_then_settles_via_fetch() -> None:
    """WS dies mid-run → status=reconnecting → one retry → final via GET /api/jobs/{id}."""
    script = _harness("""
        const fetchFn = async (url) => {
            return {
                ok: true,
                json: async () => ({
                    id: 'job-reco',
                    status: 'succeeded',
                    error: null,
                    tables_done: 1,
                    tables_total: 1,
                    total_rows: 10,
                }),
            };
        };
        const c = mkComponent({fetchFn});
        c.connect('job-reco');
        FakeWS.last.push({phase: 'table_start', table: 'a', tables_total: 1});
        // First socket dies before the terminal frame.
        FakeWS.last.drop();
        // Allow microtasks to flush so the reconnect fires synchronously.
        setTimeout(async () => {
            // After reconnect, the SECOND socket also dies before terminal.
            FakeWS.last.drop();
            // Now the factory has exhausted its single retry; it should
            // fall back to the GET /api/jobs/{id} settle path.
            await new Promise(r => setTimeout(r, 10));
            console.log(JSON.stringify({
                wsCount: FakeWS.instances.length,
                status: c.status,
                summary: c.summary,
            }));
        }, 10);
    """)
    out = _run_node(script)
    assert out["wsCount"] == 2, "exactly one reconnect attempt (2 socket instances)"
    assert out["status"] == "succeeded"
    assert "succeeded" in out["summary"].lower()


def test_handles_one_hundred_tables_without_realloc() -> None:
    """100 ``table_start`` frames produce 100 stable keys with no duplicate entries."""
    script = _harness("""
        const c = mkComponent();
        c.connect('job-100');
        for (let i = 0; i < 100; i++) {
            FakeWS.last.push({phase: 'table_start', table: 't' + i, tables_total: 100});
        }
        const names = c.tables.map(t => t.name);
        const unique = new Set(names);
        console.log(JSON.stringify({
            count: names.length,
            unique: unique.size,
            firstKey: c.tables[0].name,
            lastKey: c.tables[99].name,
        }));
    """)
    out = _run_node(script)
    assert out == {"count": 100, "unique": 100, "firstKey": "t0", "lastKey": "t99"}


def test_unknown_phase_is_ignored_safely() -> None:
    """Forward compatibility: unexpected phases must not crash the factory."""
    script = _harness("""
        const c = mkComponent();
        c.connect('job-unk');
        FakeWS.last.push({phase: 'mystery', table: 'x'});
        console.log(JSON.stringify({status: c.status, tableCount: c.tables.length}));
    """)
    out = _run_node(script)
    # Unknown phase doesn't add a table; status stays running (connect set it).
    assert out["tableCount"] == 0
    assert out["status"] == "running"
