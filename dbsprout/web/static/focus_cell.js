/* DBSprout Studio — violation drill-down → focus offending table/column (S-135)
 *
 * Wires the integrity-report panel (S-133) to the spec grid (S-118) and the
 * preview grid (S-147) so that clicking a violation row jumps the user to
 * the matching cell.
 *
 * Two public functions
 * --------------------
 *   * ``attachFocusCell(root, opts?)`` — find ``.violation-row[data-table]``
 *     descendants of ``root`` and make them clickable / keyboard-activatable.
 *     Each activation dispatches a ``studio:focus-cell`` window ``CustomEvent``
 *     with ``{table, column?, row?}`` in ``detail``. Empty ``data-row`` /
 *     ``data-column`` attributes are coerced to *omitted* (not the empty
 *     string) so listeners can distinguish "no column" from "".
 *
 *   * ``installFocusCell(root, opts?)`` — install a ``studio:focus-cell``
 *     window listener that mutates *root*: matching ``[data-table]`` section
 *     gets a ``[data-column="<t>.<c>"]`` row scrolled into view, flagged
 *     ``data-focused="true"`` and tagged with the ``studio-focus-flash`` class
 *     for ~3 s. Re-installing on the same root is idempotent — the previous
 *     listener is removed first (HTMX-swap-safe).
 *
 * The module is **transport-agnostic** — it accepts injectable hooks via the
 * ``opts`` bag: ``_now``, ``_setTimeout``, ``_clearTimeout``, ``_window``.
 * In production these default to the global ``Date.now`` / ``setTimeout`` /
 * ``clearTimeout`` / ``window`` so callers can simply pass ``{}``; tests
 * drive a fake clock + a stub ``window`` (see ``test_focus_cell_js.py``).
 *
 * Dual-loadable file (mirrors ``studio_console.js``):
 *   - In a browser: a ``<script>`` tag runs it and exposes
 *     ``window.attachFocusCell`` / ``window.installFocusCell``.
 *   - Under Node (test harness): ``require(...)`` resolves CommonJS exports
 *     ``module.exports.attachFocusCell`` / ``installFocusCell``.
 */
(function (root) {
  "use strict";

  // Flash duration in ms — long enough to draw the eye, short enough not to
  // distract from the next interaction.
  var FLASH_MS = 3000;

  // Selector used by attachFocusCell to find violation rows. The class is
  // applied by ``_validate_panel.html`` (S-133 + S-135 a11y attrs).
  var VIOLATION_SELECTOR = ".violation-row";

  // Marker key planted on the listener-root so ``installFocusCell`` is
  // idempotent across HTMX swaps.
  var INSTALLED_MARK = "__focusCellInstalled";

  function resolveWindow(opts) {
    if (opts && opts._window) return opts._window;
    if (typeof window !== "undefined") return window;
    return root;
  }

  function resolveNow(opts) {
    if (opts && typeof opts._now === "function") return opts._now;
    return function () { return Date.now(); };
  }

  function resolveSetTimeout(opts) {
    if (opts && typeof opts._setTimeout === "function") return opts._setTimeout;
    return function (cb, ms) { return setTimeout(cb, ms); };
  }

  function resolveClearTimeout(opts) {
    if (opts && typeof opts._clearTimeout === "function") return opts._clearTimeout;
    return function (id) { clearTimeout(id); };
  }

  // ── shared helpers ───────────────────────────────────────────────────

  /* Coerce a raw ``dataset`` payload into the ``studio:focus-cell`` detail
   * shape. ``data-row`` may be empty when the violation isn't tied to a
   * specific row — represent that as an *omitted* key, not the literal "".
   * ``row`` is also parsed as an integer when numeric (so listeners can do
   * a strict === compare against another integer index).
   */
  function coerceDetail(dataset) {
    var detail = { table: dataset.table };
    if (dataset.column != null && dataset.column !== "") {
      detail.column = dataset.column;
    }
    if (dataset.row != null && dataset.row !== "") {
      var n = Number(dataset.row);
      detail.row = Number.isFinite(n) && String(n) === String(dataset.row) ? n : dataset.row;
    }
    return detail;
  }

  // ── attachFocusCell: dispatch side ──────────────────────────────────

  function attachFocusCell(rootEl, opts) {
    if (!rootEl) return;
    var win = resolveWindow(opts);
    // Two-step: ``querySelectorAll(".violation-row")`` then filter on
    // ``data-table`` presence. Avoids relying on compound CSS selectors
    // so the module stays usable under minimal DOM stubs (tests).
    var candidates = rootEl.querySelectorAll(VIOLATION_SELECTOR);
    for (var i = 0; i < candidates.length; i++) {
      var row = candidates[i];
      if (!row.hasAttribute("data-table")) continue;
      // Keyboard a11y — make the row focusable + announce as a button.
      if (!row.hasAttribute("tabindex")) {
        row.setAttribute("tabindex", "0");
      }
      if (!row.hasAttribute("role")) {
        row.setAttribute("role", "button");
      }
      if (!row.hasAttribute("aria-label")) {
        var t = row.getAttribute("data-table") || "";
        var c = row.getAttribute("data-column") || "";
        var label = c
          ? "Focus " + t + "." + c
          : "Focus " + t;
        row.setAttribute("aria-label", label);
      }
      wireRow(row, win);
    }
  }

  function wireRow(row, win) {
    function fire() {
      var ev = new win.CustomEvent("studio:focus-cell", { detail: coerceDetail(row.dataset) });
      win.dispatchEvent(ev);
    }
    row.addEventListener("click", function () {
      fire();
    });
    row.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" || ev.key === " ") {
        if (typeof ev.preventDefault === "function") ev.preventDefault();
        fire();
      }
    });
  }

  // ── installFocusCell: listener side ────────────────────────────────

  function installFocusCell(rootEl, opts) {
    if (!rootEl) return;
    var win = resolveWindow(opts);
    var now = resolveNow(opts);
    var sTimeout = resolveSetTimeout(opts);
    var cTimeout = resolveClearTimeout(opts);

    // Tear down any previous listener on this root so HTMX swaps don't stack.
    var prev = rootEl[INSTALLED_MARK];
    if (prev && typeof prev.detach === "function") {
      prev.detach();
    }

    var pendingTimer = null;

    function handler(ev) {
      var detail = (ev && ev.detail) || {};
      if (!detail.table) return;
      focusInSpecGrid(rootEl, detail);
      focusInPreviewGrid(rootEl, detail);
    }

    function focusInSpecGrid(grid, detail) {
      // Only act when this root *is* the spec grid (or contains a spec-grid section).
      var section = grid.querySelector("[data-table=\"" + cssEscape(detail.table) + "\"]");
      if (!section) return;
      if (!detail.column) {
        // Table-level focus: scroll the section itself.
        section.scrollIntoView();
        return;
      }
      var targetCol = detail.table + "." + detail.column;
      var row = section.querySelector("[data-column=\"" + cssEscape(targetCol) + "\"]");
      if (!row) return;
      row.scrollIntoView();
      row.setAttribute("data-focused", "true");
      row.classList.add("studio-focus-flash");
      if (pendingTimer !== null) cTimeout(pendingTimer);
      pendingTimer = sTimeout(function () {
        row.classList.remove("studio-focus-flash");
        pendingTimer = null;
      }, FLASH_MS);
    }

    function focusInPreviewGrid(grid, detail) {
      if (detail.row == null) return;
      var previewContainer = grid.querySelector(
        "[data-preview-table=\"" + cssEscape(detail.table) + "\"]"
      );
      if (!previewContainer) {
        // Maybe this root *is* the preview container.
        if (grid.getAttribute && grid.getAttribute("data-preview-table") === detail.table) {
          previewContainer = grid;
        }
      }
      if (!previewContainer) return;
      var idx = String(detail.row);
      var previewRow = previewContainer.querySelector(
        "[data-row-index=\"" + cssEscape(idx) + "\"]"
      );
      if (!previewRow) return;
      previewRow.scrollIntoView();
    }

    win.addEventListener("studio:focus-cell", handler);

    // Mark the install so we can replace it on re-install.
    rootEl[INSTALLED_MARK] = {
      detach: function () {
        win.removeEventListener("studio:focus-cell", handler);
        if (pendingTimer !== null) cTimeout(pendingTimer);
      },
    };

    // Used by callers + tests that want to detach explicitly.
    return rootEl[INSTALLED_MARK].detach;
  }

  // Light CSS.escape polyfill — values come from server-rendered template
  // tokens (already trusted) but we still guard quoted attribute selectors.
  function cssEscape(value) {
    return String(value).replace(/(["\\])/g, "\\$1");
  }

  // ── exports ──────────────────────────────────────────────────────────
  root.attachFocusCell = attachFocusCell;
  root.installFocusCell = installFocusCell;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { attachFocusCell: attachFocusCell, installFocusCell: installFocusCell };
  }
})(typeof window !== "undefined" ? window : (typeof global !== "undefined" ? global : this));
