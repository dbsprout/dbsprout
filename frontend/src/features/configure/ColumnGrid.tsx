import { useEffect, useRef, useState, type ChangeEvent, type CSSProperties } from "react";
import { observeElementRect, useVirtualizer, type Virtualizer } from "@tanstack/react-virtual";
import type { GeneratorConfig, GeneratorMethod, TableSpec } from "../../api/types";
import { generatorOptions } from "./columnDtype";
import { genLabel } from "./genLabel";

interface ColumnGridProps {
  spec: TableSpec;
  methods: GeneratorMethod[];
  /** First row of the table preview, used for the live "sample" column. */
  previewRow: Record<string, unknown> | null;
  /**
   * Raw SQL type per column name (e.g. `{ email: "VARCHAR(255)" }`). When given,
   * each column's generator dropdown is filtered to dtype-compatible generators
   * (P4-1). Optional so callers that don't have schema types stay unfiltered.
   */
  columnTypes?: Record<string, string>;
  /**
   * Cross-panel drill target (P4-7): the column to scroll into view and
   * highlight (e.g. from a Validate violation). No-op when null/absent or when
   * the column no longer exists in `spec`.
   */
  focusColumn?: string | null;
  /** Focus a column in the Inspector. */
  onSelect: (column: string) => void;
  /** Persist a new generator config for a column. */
  onGeneratorChange: (column: string, cfg: GeneratorConfig) => void;
}

/**
 * Row index of a cross-panel focus target within the ordered column names, or -1
 * when the target is null/absent (so callers treat it as a no-op). Pure + exported
 * for unit testing the drill-to-cell logic without the jsdom-unfriendly virtualizer.
 */
export function focusRowIndex(
  order: readonly string[],
  focusColumn: string | null | undefined,
): number {
  return focusColumn == null ? -1 : order.indexOf(focusColumn);
}

/** Estimated row height (px); rows are uniform single-line cells. */
const ROW_HEIGHT = 40;
/** Bounded viewport height (px) so only visible rows mount on wide tables. */
const VIEWPORT_HEIGHT = 480;

/** Five equal-width cells laid out as a flex row so columns line up without table layout. */
const ROW_STYLE: CSSProperties = { display: "flex", width: "100%", height: ROW_HEIGHT, alignItems: "center" };
const CELL_STYLE: CSSProperties = { flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" };

/**
 * The scroll container has a CSS-fixed height (`VIEWPORT_HEIGHT`). Floor the observed
 * viewport height to it so the window is computed even where the layout engine reports
 * 0 height before measuring (e.g. jsdom, or the first paint). In a real browser the
 * measured height already equals the fixed height, so this is a no-op there.
 */
function observeFixedRect(
  instance: Virtualizer<HTMLTableSectionElement, Element>,
  cb: (rect: { width: number; height: number }) => void,
): void | (() => void) {
  return observeElementRect(instance, (rect) =>
    cb({ width: rect.width, height: Math.max(rect.height, VIEWPORT_HEIGHT) }),
  );
}

/**
 * Per-column grid for the active table: name · generator ▾ · params · sample · key.
 *
 * The body is virtualized with `@tanstack/react-virtual`: only the rows within the
 * scroll viewport (plus overscan) are mounted, so wide tables (100s of columns) stay
 * responsive. The `<tbody>` is a native `overflow:auto` scroll container, so keyboard
 * scrolling works, and each row keeps its focusable `<button>`/`<select>`.
 */
export function ColumnGrid({
  spec,
  methods,
  previewRow,
  columnTypes,
  focusColumn,
  onSelect,
  onGeneratorChange,
}: ColumnGridProps) {
  const entries = Object.entries(spec.columns);
  const parentRef = useRef<HTMLTableSectionElement>(null);
  // P4-1: escape hatch — show every generator regardless of column dtype.
  const [showAll, setShowAll] = useState(false);

  const virtualizer = useVirtualizer({
    count: entries.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 8,
    // Seed the viewport with the known fixed height so the window is computed even
    // before the browser measures the scroll element (and in jsdom, which reports 0).
    initialRect: { width: 0, height: VIEWPORT_HEIGHT },
    observeElementRect: observeFixedRect,
  });

  // P4-7: cross-panel drill — index of the focused column (-1 when null/absent).
  const focusIndex = focusRowIndex(
    entries.map(([name]) => name),
    focusColumn,
  );

  // Scroll the focused row into the virtualized window when the target changes.
  useEffect(() => {
    if (focusIndex >= 0) {
      virtualizer.scrollToIndex(focusIndex, { align: "center" });
    }
  }, [focusIndex, virtualizer]);

  function handleChange(
    column: string,
    cfg: GeneratorConfig,
    e: ChangeEvent<HTMLSelectElement>,
  ) {
    const [provider, method = null] = e.target.value.split("/");
    onGeneratorChange(column, { ...cfg, provider, method });
  }

  return (
    <div className="db-subsection">
      <label className="mb-2 flex items-center gap-2 text-sm text-slate-600">
        <input
          type="checkbox"
          aria-label="show all generators"
          checked={showAll}
          onChange={(e) => setShowAll(e.target.checked)}
          className="accent-accent-600"
        />
        show all generators
      </label>
      <table
        aria-label={`columns of ${spec.table_name}`}
        style={{ display: "block", width: "100%" }}
        className="rounded-md border border-slate-200 text-sm"
      >
        <thead style={{ display: "block" }} className="bg-slate-100">
        <tr style={ROW_STYLE}>
          <th style={CELL_STYLE} className="px-2 text-xs font-semibold uppercase tracking-wide text-slate-500">name</th>
          <th style={CELL_STYLE} className="px-2 text-xs font-semibold uppercase tracking-wide text-slate-500">generator</th>
          <th style={CELL_STYLE} className="px-2 text-xs font-semibold uppercase tracking-wide text-slate-500">params</th>
          <th style={CELL_STYLE} className="px-2 text-xs font-semibold uppercase tracking-wide text-slate-500">sample</th>
          <th style={CELL_STYLE} className="px-2 text-xs font-semibold uppercase tracking-wide text-slate-500">key</th>
        </tr>
      </thead>
      <tbody
        ref={parentRef}
        style={{ display: "block", position: "relative", height: VIEWPORT_HEIGHT, overflow: "auto" }}
        className="bg-white"
      >
        <tr style={{ display: "block", height: virtualizer.getTotalSize(), position: "relative" }}>
          <td style={{ display: "block", padding: 0, border: 0 }}>
            {virtualizer.getVirtualItems().map((virtualRow) => {
              const [name, cfg] = entries[virtualRow.index];
              const current = genLabel(cfg.provider, cfg.method);
              const options = generatorOptions(methods, columnTypes?.[name], showAll, current);
              const sample = previewRow ? String(previewRow[name] ?? "") : "—";
              const paramSummary = Object.keys(cfg.params).length ? JSON.stringify(cfg.params) : "—";
              const focused = virtualRow.index === focusIndex;
              return (
                <div
                  key={name}
                  data-focused={focused}
                  className="border-b border-slate-100 px-2 hover:bg-slate-50"
                  style={{
                    ...ROW_STYLE,
                    position: "absolute",
                    top: 0,
                    left: 0,
                    transform: `translateY(${virtualRow.start}px)`,
                    // P4-7: highlight the drilled-into row.
                    outline: focused ? "2px solid #2563eb" : undefined,
                    background: focused ? "#eff6ff" : undefined,
                  }}
                >
                  <span style={CELL_STYLE} className="px-1">
                    <button
                      type="button"
                      aria-label={`inspect ${name}`}
                      onClick={() => onSelect(name)}
                      className="font-mono text-[13px] font-medium text-accent-700 hover:underline"
                    >
                      {name}
                    </button>
                  </span>
                  <span style={CELL_STYLE} className="px-1">
                    <select
                      aria-label={`generator for ${name}`}
                      value={current}
                      onChange={(e) => handleChange(name, cfg, e)}
                      className="w-full rounded border border-slate-300 bg-white px-1 py-0.5 text-xs outline-none focus:border-accent-500 focus:ring-1 focus:ring-accent-500/40"
                    >
                      {options.map((o) => (
                        <option key={o.value} value={o.value}>
                          {o.label}
                        </option>
                      ))}
                    </select>
                  </span>
                  <span style={CELL_STYLE} className="px-1 font-mono text-xs text-slate-500">{paramSummary}</span>
                  <span style={CELL_STYLE} className="px-1 font-mono text-xs text-slate-700">{sample}</span>
                  <span style={CELL_STYLE} className="px-1">
                    {cfg.unique ? <span className="db-badge bg-violet-100 text-violet-700">unique</span> : ""}
                  </span>
                </div>
              );
            })}
          </td>
        </tr>
      </tbody>
      </table>
    </div>
  );
}
