import { useRef, type ChangeEvent, type CSSProperties } from "react";
import { observeElementRect, useVirtualizer, type Virtualizer } from "@tanstack/react-virtual";
import type { GeneratorConfig, GeneratorMethod, TableSpec } from "../../api/types";
import { genLabel } from "./genLabel";

interface ColumnGridProps {
  spec: TableSpec;
  methods: GeneratorMethod[];
  /** First row of the table preview, used for the live "sample" column. */
  previewRow: Record<string, unknown> | null;
  /** Focus a column in the Inspector. */
  onSelect: (column: string) => void;
  /** Persist a new generator config for a column. */
  onGeneratorChange: (column: string, cfg: GeneratorConfig) => void;
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
  onSelect,
  onGeneratorChange,
}: ColumnGridProps) {
  const options = methods.map((m) => genLabel(m.provider, m.method));
  const entries = Object.entries(spec.columns);
  const parentRef = useRef<HTMLTableSectionElement>(null);

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

  function handleChange(
    column: string,
    cfg: GeneratorConfig,
    e: ChangeEvent<HTMLSelectElement>,
  ) {
    const [provider, method = null] = e.target.value.split("/");
    onGeneratorChange(column, { ...cfg, provider, method });
  }

  return (
    <table aria-label={`columns of ${spec.table_name}`} style={{ display: "block", width: "100%" }}>
      <thead style={{ display: "block" }}>
        <tr style={ROW_STYLE}>
          <th style={CELL_STYLE}>name</th>
          <th style={CELL_STYLE}>generator</th>
          <th style={CELL_STYLE}>params</th>
          <th style={CELL_STYLE}>sample</th>
          <th style={CELL_STYLE}>key</th>
        </tr>
      </thead>
      <tbody
        ref={parentRef}
        style={{ display: "block", position: "relative", height: VIEWPORT_HEIGHT, overflow: "auto" }}
      >
        <tr style={{ display: "block", height: virtualizer.getTotalSize(), position: "relative" }}>
          <td style={{ display: "block", padding: 0, border: 0 }}>
            {virtualizer.getVirtualItems().map((virtualRow) => {
              const [name, cfg] = entries[virtualRow.index];
              const current = genLabel(cfg.provider, cfg.method);
              const sample = previewRow ? String(previewRow[name] ?? "") : "—";
              const paramSummary = Object.keys(cfg.params).length ? JSON.stringify(cfg.params) : "—";
              return (
                <div
                  key={name}
                  style={{
                    ...ROW_STYLE,
                    position: "absolute",
                    top: 0,
                    left: 0,
                    transform: `translateY(${virtualRow.start}px)`,
                  }}
                >
                  <span style={CELL_STYLE}>
                    <button type="button" aria-label={`inspect ${name}`} onClick={() => onSelect(name)}>
                      {name}
                    </button>
                  </span>
                  <span style={CELL_STYLE}>
                    <select
                      aria-label={`generator for ${name}`}
                      value={current}
                      onChange={(e) => handleChange(name, cfg, e)}
                    >
                      {!options.includes(current) && <option value={current}>{current}</option>}
                      {options.map((o) => (
                        <option key={o} value={o}>
                          {o}
                        </option>
                      ))}
                    </select>
                  </span>
                  <span style={CELL_STYLE}>{paramSummary}</span>
                  <span style={CELL_STYLE}>{sample}</span>
                  <span style={CELL_STYLE}>{cfg.unique ? "unique" : ""}</span>
                </div>
              );
            })}
          </td>
        </tr>
      </tbody>
    </table>
  );
}
