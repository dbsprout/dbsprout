/* DBSprout Studio — per-column method swap → instant re-preview (S-132)
 *
 * Wires the S-120 method-picker swap to the S-131 regenerate route and
 * the S-147 preview route. The flow on a single column method swap is:
 *
 *   1. The picker PUTs the new GeneratorConfig (S-119) and ``outerHTML``-
 *      swaps the row, then dispatches a ``studio:row-rerendered`` window
 *      ``CustomEvent`` with ``{table, column, ...}`` in ``detail``.
 *   2. This module listens for that event, **debounces** per-column
 *      (~150 ms — coalesces rapid swaps to the last selection), then
 *      POSTs ``/api/regenerate`` with ``{table, column, reroll: <nonce>}``.
 *      The nonce bumps once per actual fire so successive swaps produce
 *      fresh draws.
 *   3. On regen success: GETs ``/api/preview/{table}?limit=N`` and
 *      dispatches ``studio:preview-updated`` with ``{table, column,
 *      values, total}``. ``values`` is the slice of the requested
 *      column from each row — keeping the seam tight so any future
 *      preview-grid UI can subscribe without re-implementing the fetch.
 *   4. On any failure: dispatches ``studio:preview-error`` with
 *      ``{table, column, message}``. The spec is NOT reverted — the
 *      PUT already persisted server-side, and the user can retry or
 *      pick a different method without losing their change.
 *
 * The affected ``[data-column="<t>.<c>"]`` row carries a
 * ``data-regenerating="true"`` attribute while a regen is in flight (a
 * CSS hook for spinner/opacity), removed on either success or failure.
 *
 * Untouched-row invariant
 * -----------------------
 * Only the row matching the event's ``{table, column}`` is mutated.
 * Sibling rows — even in the same table — are not touched by this
 * module, satisfying the S-132 "no regression on rows untouched by the
 * swap" AC.
 *
 * Test surface
 * ------------
 * Injectable hooks via the ``opts`` bag:
 *
 *   * ``_fetch`` — replaces ``window.fetch``.
 *   * ``_setTimeout`` / ``_clearTimeout`` — drive the debounce.
 *   * ``_window`` — replaces ``window`` (event bus).
 *   * ``_document`` — replaces ``document`` (DOM lookup).
 *   * ``debounceMs`` — overrides the 150 ms default.
 *   * ``previewLimit`` — overrides the ``?limit=`` query param (100 default).
 *
 * Dual-loadable file (mirrors ``focus_cell.js``):
 *   - In a browser: ``window.attachSpecGridRerender`` is set.
 *   - Under Node (test harness): ``module.exports.attachSpecGridRerender``.
 */
(function (root) {
  "use strict";

  // Default debounce window — short enough to feel instant, long enough
  // to coalesce a rapid double-click on different method buttons.
  var DEFAULT_DEBOUNCE_MS = 150;

  // Default sample size on the preview re-fetch. Mirrors the S-147
  // ``GET /api/preview/{table}`` default and stays well under the 1000-
  // row server cap.
  var DEFAULT_PREVIEW_LIMIT = 100;

  function resolveWindow(opts) {
    if (opts && opts._window) return opts._window;
    if (typeof window !== "undefined") return window;
    return root;
  }

  function resolveDocument(opts) {
    if (opts && opts._document) return opts._document;
    if (typeof document !== "undefined") return document;
    return null;
  }

  function resolveFetch(opts) {
    if (opts && typeof opts._fetch === "function") return opts._fetch;
    if (typeof fetch !== "undefined") return fetch;
    // No fetch available — return a function that always rejects so the
    // failure path is exercised cleanly.
    return function () {
      return Promise.reject(new Error("fetch is not available"));
    };
  }

  function resolveSetTimeout(opts) {
    if (opts && typeof opts._setTimeout === "function") return opts._setTimeout;
    return function (cb, ms) { return setTimeout(cb, ms); };
  }

  function resolveClearTimeout(opts) {
    if (opts && typeof opts._clearTimeout === "function") return opts._clearTimeout;
    return function (id) { return clearTimeout(id); };
  }

  // Build the CustomEvent constructor in a way that works under both
  // the real browser and the Node test harness (which supplies a stub
  // ``CustomEvent`` on the injected ``_window``).
  function makeCustomEvent(win, type, detail) {
    var Ctor = win && win.CustomEvent ? win.CustomEvent :
      (typeof CustomEvent !== "undefined" ? CustomEvent : null);
    if (Ctor) {
      return new Ctor(type, { detail: detail });
    }
    // Last-ditch fallback — bare object that still has ``type`` and
    // ``detail`` so listeners can read both. Should never hit in
    // practice (every supported environment has CustomEvent).
    return { type: type, detail: detail };
  }

  function attachSpecGridRerender(opts) {
    var options = opts || {};
    var win = resolveWindow(options);
    var doc = resolveDocument(options);
    var fetchFn = resolveFetch(options);
    var schedule = resolveSetTimeout(options);
    var cancel = resolveClearTimeout(options);
    var debounceMs = typeof options.debounceMs === "number" ? options.debounceMs : DEFAULT_DEBOUNCE_MS;
    var previewLimit = typeof options.previewLimit === "number" ? options.previewLimit : DEFAULT_PREVIEW_LIMIT;

    // Per-column key → currently-scheduled timeout (undefined when none
    // is pending). Used to coalesce rapid swaps trailing-edge to the
    // last selection.
    var pending = {};

    // Per-column key → next reroll nonce to send. We bump *once per
    // actual fire* (after the debounce settles), so coalesced events
    // share a nonce but two non-coalesced calls send incrementing
    // integers.
    var nonces = {};

    function key(detail) {
      return String(detail.table) + "." + String(detail.column);
    }

    function findRow(detail) {
      if (!doc || typeof doc.querySelector !== "function") return null;
      // ``data-column`` carries ``<table>.<column>`` on the spec-grid row
      // wrapper. We don't escape because the values come from a Pydantic
      // model with strict identifier validation.
      return doc.querySelector('[data-column="' + key(detail) + '"]');
    }

    function setLoading(row, value) {
      if (!row) return;
      if (value) {
        if (typeof row.setAttribute === "function") {
          row.setAttribute("data-regenerating", "true");
        }
      } else if (typeof row.removeAttribute === "function") {
        row.removeAttribute("data-regenerating");
      }
    }

    function dispatchError(table, column, message) {
      win.dispatchEvent(makeCustomEvent(win, "studio:preview-error", {
        table: table,
        column: column,
        message: message,
      }));
    }

    function dispatchUpdated(table, column, values, total) {
      win.dispatchEvent(makeCustomEvent(win, "studio:preview-updated", {
        table: table,
        column: column,
        values: values,
        total: total,
      }));
    }

    function readErrorMessage(resp) {
      // Best-effort text → never throw. We only use the message for the
      // inline error event, so falling back to a generic line is fine.
      if (resp && typeof resp.text === "function") {
        return resp.text().then(function (txt) {
          if (txt) return txt;
          return "HTTP " + (resp.status || "error");
        }).catch(function () {
          return "HTTP " + (resp.status || "error");
        });
      }
      return Promise.resolve("HTTP " + ((resp && resp.status) || "error"));
    }

    function extractColumnSlice(rows, column) {
      var values = [];
      if (!Array.isArray(rows)) return values;
      for (var i = 0; i < rows.length; i++) {
        var r = rows[i];
        if (r && Object.prototype.hasOwnProperty.call(r, column)) {
          values.push(r[column]);
        } else {
          // Defensive: missing column on a row → push null so the
          // ``values`` array stays positionally aligned with ``rows``.
          values.push(null);
        }
      }
      return values;
    }

    function fireRegen(table, column) {
      var k = table + "." + column;
      nonces[k] = (nonces[k] || 0) + 1;
      var reroll = nonces[k];
      var row = findRow({ table: table, column: column });
      setLoading(row, true);

      var regenBody = JSON.stringify({ table: table, column: column, reroll: reroll });
      var regenInit = {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: regenBody,
      };

      fetchFn("/api/regenerate", regenInit).then(function (resp) {
        if (!resp || !resp.ok) {
          return readErrorMessage(resp).then(function (msg) {
            setLoading(row, false);
            dispatchError(table, column, "Regenerate failed: " + msg);
          });
        }
        // Regen ok → fetch the bounded preview slice.
        var previewUrl = "/api/preview/" + encodeURIComponent(table) +
          "?limit=" + encodeURIComponent(previewLimit);
        return fetchFn(previewUrl, { method: "GET", headers: { Accept: "application/json" } })
          .then(function (presp) {
            if (!presp || !presp.ok) {
              return readErrorMessage(presp).then(function (msg) {
                setLoading(row, false);
                dispatchError(table, column, "Preview fetch failed: " + msg);
              });
            }
            return presp.json().then(function (payload) {
              setLoading(row, false);
              var rows = (payload && payload.rows) || [];
              var total = (payload && typeof payload.total === "number") ? payload.total : rows.length;
              dispatchUpdated(table, column, extractColumnSlice(rows, column), total);
            });
          });
      }).catch(function (err) {
        setLoading(row, false);
        dispatchError(table, column, "Regenerate failed: " + (err && err.message ? err.message : String(err)));
      });
    }

    function onRowRerendered(ev) {
      var detail = ev && ev.detail;
      if (!detail) return;
      var table = detail.table;
      var column = detail.column;
      if (!table || !column) return;
      var k = key(detail);
      // Trailing-edge debounce per column key. Cancel any pending one
      // and schedule a fresh fire — the *last* event in the window wins.
      if (pending[k] !== undefined) {
        cancel(pending[k]);
      }
      pending[k] = schedule(function () {
        delete pending[k];
        fireRegen(table, column);
      }, debounceMs);
    }

    win.addEventListener("studio:row-rerendered", onRowRerendered);

    // Detach is rarely needed — the studio page lives until full
    // reload — but expose one anyway for tests / future HTMX swaps.
    return {
      detach: function () {
        win.removeEventListener("studio:row-rerendered", onRowRerendered);
      },
    };
  }

  // ── exports ──────────────────────────────────────────────────────────
  root.attachSpecGridRerender = attachSpecGridRerender;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { attachSpecGridRerender: attachSpecGridRerender };
  }
})(typeof window !== "undefined" ? window : (typeof global !== "undefined" ? global : this));
