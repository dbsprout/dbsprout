/* DBSprout Studio — live progress console (S-125)
 *
 * Alpine.js x-data factory that:
 *   - listens for a 'studio:job-start' window event with {jobId}
 *   - opens a WebSocket to /ws/jobs/{jobId} (the S-109 endpoint)
 *   - folds incoming ProgressEvent frames into a small reactive state object
 *   - renders per-table progress bars + an overall bar + rows/sec + a status
 *     badge + a terminal summary line (the markup lives in
 *     templates/_studio_console.html)
 *
 * Design notes
 * ============
 * * The factory is **transport-agnostic** — it accepts injectable
 *   ``_wsFactory(url) -> WebSocketLike`` and ``_fetchFn(url) -> Promise<Response>``
 *   hooks. In production these default to ``window.WebSocket`` and
 *   ``window.fetch``; tests drive a ``FakeWS`` (see ``test_studio_console_js.py``).
 * * State is **plain JS** (arrays + numbers) so Alpine can reactively render
 *   it; the per-table list is keyed by table name so Alpine ``x-for`` reuses
 *   DOM nodes across frames (no full re-render on every event).
 * * Rolling rows/sec uses a 5-sample window of ``(ts, total_rows)`` pairs,
 *   smoothing jitter from per-table batch boundaries.
 * * On mid-run WS disconnect the factory retries **exactly once**, then
 *   settles the final state via ``GET /api/jobs/{id}`` so the user still sees
 *   the terminal outcome even if both sockets dropped before the terminal
 *   frame arrived.
 *
 * This file is dual-loadable:
 *   - In a browser: a ``<script>`` tag executes it and Alpine reads
 *     ``window.studioConsole`` when it processes ``x-data="studioConsole()"``.
 *   - Under Node (the test harness): ``require(...)`` resolves it as a
 *     CommonJS module and reads ``module.exports.studioConsole``.
 */
(function (root) {
  "use strict";

  // Statuses surfaced by the badge. Kept in sync with the dashboard's
  // intuitive vocabulary; "running" / "reconnecting" are pre-terminal.
  var TERMINAL = { succeeded: 1, failed: 1, cancelled: 1 };

  function studioConsole() {
    return {
      // ── reactive state (read by the Alpine template) ─────────────────
      status: "idle",
      tables: [],               // [{name, rows_done, rows_total, percent}]
      _tablesIndex: {},          // name -> index into ``tables`` (stable key)
      overall: 0,                // 0..100
      rowsPerSec: 0,
      tablesDone: 0,
      tablesTotal: 0,
      totalRows: 0,
      summary: "",
      frozen: false,
      error: null,
      jobId: null,
      _ws: null,
      _retried: false,
      _samples: [],              // last 5 (ts_ms, total_rows) pairs
      _settled: false,

      // ── hooks (defaults; tests override) ──────────────────────────────
      _wsFactory: function (url) {
        return new root.WebSocket(url);
      },
      _fetchFn: function (url) {
        return root.fetch(url);
      },
      _now: function () {
        return Date.now();
      },

      // ── Alpine lifecycle: wire the start event listener ──────────────
      init: function () {
        var self = this;
        if (root.addEventListener) {
          root.addEventListener("studio:job-start", function (ev) {
            var detail = ev && ev.detail ? ev.detail : {};
            if (detail.jobId) self.connect(detail.jobId);
          });
        }
      },

      // ── public: connect to /ws/jobs/{jobId} ──────────────────────────
      connect: function (jobId) {
        if (!jobId) return;
        this.jobId = jobId;
        this.status = "running";
        this.frozen = false;
        this._retried = false;
        this._settled = false;
        this._openSocket();
      },

      _openSocket: function () {
        var self = this;
        var url = this._wsUrl(this.jobId);
        var ws;
        try {
          ws = this._wsFactory(url);
        } catch (e) {
          this._settleViaPoll();
          return;
        }
        this._ws = ws;
        ws.onmessage = function (msg) {
          self._onMessage(msg);
        };
        ws.onclose = function () {
          self._onClose();
        };
        ws.onerror = function () {
          // Errors are surfaced via the subsequent onclose; nothing to do.
        };
      },

      _wsUrl: function (jobId) {
        // Best-effort URL derivation; in tests the wsFactory ignores it.
        var loc = (root.location || {});
        var scheme = loc.protocol === "https:" ? "wss:" : "ws:";
        var host = loc.host || "localhost";
        return scheme + "//" + host + "/ws/jobs/" + encodeURIComponent(jobId);
      },

      // ── frame routing ─────────────────────────────────────────────────
      _onMessage: function (msg) {
        if (this.frozen) return;
        var frame;
        try {
          frame = typeof msg.data === "string" ? JSON.parse(msg.data) : msg.data;
        } catch (e) {
          return;
        }
        if (!frame || !frame.phase) return;
        switch (frame.phase) {
          case "table_start":
            this._applyTableStart(frame);
            break;
          case "table_done":
            this._applyTableDone(frame);
            break;
          case "terminal":
            this._applyTerminal(frame);
            break;
          default:
            // Forward-compat: ignore unknown phases without crashing.
            return;
        }
      },

      _applyTableStart: function (frame) {
        if (frame.tables_total) this.tablesTotal = frame.tables_total;
        if (frame.table) this._ensureTable(frame.table);
        this._recordSample(frame.total_rows || this.totalRows);
        this._recomputeOverall();
      },

      _applyTableDone: function (frame) {
        if (frame.tables_total) this.tablesTotal = frame.tables_total;
        if (typeof frame.tables_done === "number") this.tablesDone = frame.tables_done;
        if (typeof frame.total_rows === "number") this.totalRows = frame.total_rows;
        if (frame.table) {
          var entry = this._ensureTable(frame.table);
          entry.rows_done = frame.rows_in_table || entry.rows_done || 0;
          entry.rows_total = entry.rows_total || entry.rows_done;
          entry.percent = entry.rows_total > 0 ? 100 : 0;
        }
        this._recordSample(this.totalRows);
        this._recomputeOverall();
      },

      _applyTerminal: function (frame) {
        this.status = frame.status || "succeeded";
        this.error = frame.error || null;
        this.frozen = true;
        if (this.tablesTotal > 0 && TERMINAL[this.status]) {
          // Snap overall to 100% on success so the user sees a clean finish.
          if (this.status === "succeeded") this.overall = 100;
        }
        this.summary = this._buildSummary();
        if (this._ws && typeof this._ws.close === "function" && !this._ws.sentClose) {
          try { this._ws.close(); } catch (e) { /* noop */ }
        }
        this._settled = true;
      },

      _ensureTable: function (name) {
        var idx = this._tablesIndex[name];
        if (typeof idx === "number") return this.tables[idx];
        var entry = { name: name, rows_done: 0, rows_total: 0, percent: 0 };
        this._tablesIndex[name] = this.tables.length;
        this.tables.push(entry);
        return entry;
      },

      _recordSample: function (totalRows) {
        var ts = this._now();
        this._samples.push([ts, totalRows || 0]);
        if (this._samples.length > 5) this._samples.shift();
        this.rowsPerSec = this._computeRate();
      },

      _computeRate: function () {
        if (this._samples.length < 2) return 0;
        var first = this._samples[0];
        var last = this._samples[this._samples.length - 1];
        var dRows = last[1] - first[1];
        var dMs = last[0] - first[0];
        if (dMs <= 0 || dRows <= 0) return 0;
        return Math.round((dRows / dMs) * 1000);
      },

      _recomputeOverall: function () {
        if (this.tablesTotal > 0) {
          this.overall = Math.min(
            100,
            Math.round((this.tablesDone / this.tablesTotal) * 100)
          );
        }
      },

      // ── disconnect / reconnect / poll-fallback ────────────────────────
      _onClose: function () {
        if (this._settled || this.frozen) return;
        if (!this._retried) {
          this._retried = true;
          this.status = "reconnecting";
          this._openSocket();
          return;
        }
        // Exhausted retry → settle from the REST endpoint.
        this._settleViaPoll();
      },

      _settleViaPoll: function () {
        var self = this;
        var url = "/api/jobs/" + encodeURIComponent(this.jobId || "");
        var p;
        try {
          p = this._fetchFn(url);
        } catch (e) {
          self.status = "failed";
          self.error = String(e);
          self.summary = self._buildSummary();
          self.frozen = true;
          return;
        }
        Promise.resolve(p)
          .then(function (resp) {
            if (!resp || !resp.ok) throw new Error("HTTP " + (resp && resp.status));
            return resp.json();
          })
          .then(function (job) {
            self.status = job.status || "failed";
            self.error = job.error || null;
            if (typeof job.tables_done === "number") self.tablesDone = job.tables_done;
            if (typeof job.tables_total === "number") self.tablesTotal = job.tables_total;
            if (typeof job.total_rows === "number") self.totalRows = job.total_rows;
            self._recomputeOverall();
            self.frozen = true;
            self.summary = self._buildSummary();
            self._settled = true;
          })
          .catch(function (err) {
            self.status = "failed";
            self.error = String(err);
            self.summary = self._buildSummary();
            self.frozen = true;
            self._settled = true;
          });
      },

      // ── derived getters used by the template ──────────────────────────
      get visibleTables() {
        return this.tables;
      },

      _buildSummary: function () {
        var head;
        if (this.status === "succeeded") head = "Succeeded";
        else if (this.status === "failed") head = "Failed";
        else if (this.status === "cancelled") head = "Cancelled";
        else head = this.status || "Done";
        var line = head + " · " + this.tablesDone + "/" + this.tablesTotal + " tables · " + this.totalRows + " rows";
        if (this.error) line += " · " + this.error;
        return line;
      },
    };
  }

  // Expose under both window (browser) and module.exports (Node test harness).
  root.studioConsole = studioConsole;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { studioConsole: studioConsole };
  }
})(typeof window !== "undefined" ? window : (typeof global !== "undefined" ? global : this));
