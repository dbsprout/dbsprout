import type { ChangeEvent } from "react";
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

/** Per-column grid for the active table: name · generator ▾ · params · sample · key. */
export function ColumnGrid({
  spec,
  methods,
  previewRow,
  onSelect,
  onGeneratorChange,
}: ColumnGridProps) {
  const options = methods.map((m) => genLabel(m.provider, m.method));

  function handleChange(
    column: string,
    cfg: GeneratorConfig,
    e: ChangeEvent<HTMLSelectElement>,
  ) {
    const [provider, method = null] = e.target.value.split("/");
    onGeneratorChange(column, { ...cfg, provider, method });
  }

  return (
    <table aria-label={`columns of ${spec.table_name}`}>
      <thead>
        <tr>
          <th>name</th>
          <th>generator</th>
          <th>params</th>
          <th>sample</th>
          <th>key</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(spec.columns).map(([name, cfg]) => {
          const current = genLabel(cfg.provider, cfg.method);
          const sample = previewRow ? String(previewRow[name] ?? "") : "—";
          const paramSummary = Object.keys(cfg.params).length ? JSON.stringify(cfg.params) : "—";
          return (
            <tr key={name}>
              <td>
                <button type="button" aria-label={`inspect ${name}`} onClick={() => onSelect(name)}>
                  {name}
                </button>
              </td>
              <td>
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
              </td>
              <td>{paramSummary}</td>
              <td>{sample}</td>
              <td>{cfg.unique ? "unique" : ""}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
