import { useState } from "react";
import type { CorrelationRule } from "../../api/types";

interface CorrelationsEditorProps {
  /** Active table name (used for labelling + draft reset). */
  table: string;
  /** Column names available on the active table. */
  columns: string[];
  /** All table names in the schema (for the optional lookup_table picker). */
  tables: string[];
  /** Current correlation rules for the active table. */
  rules: CorrelationRule[];
  /** Persist the full new rule list. */
  onSave: (rules: CorrelationRule[]) => void;
}

/**
 * Edits a table's correlation rules (FK fan-out / cross-column coherence such as
 * city/state/zip lookups). Keyed on `table` so the add-rule draft resets cleanly
 * when the user switches tables.
 */
export function CorrelationsEditor(props: CorrelationsEditorProps) {
  return <Editor key={props.table} {...props} />;
}

function Editor({ table, columns, tables, rules, onSave }: CorrelationsEditorProps) {
  const [draftColumns, setDraftColumns] = useState<string[]>([]);
  const [lookupTable, setLookupTable] = useState("");
  const [strategy, setStrategy] = useState("lookup");
  const [error, setError] = useState<string | null>(null);

  function handleAdd() {
    if (draftColumns.length === 0) {
      setError("Select at least one column for the correlation.");
      return;
    }
    setError(null);
    const rule: CorrelationRule = {
      columns: draftColumns,
      lookup_table: lookupTable === "" ? null : lookupTable,
      strategy: strategy || "lookup",
    };
    onSave([...rules, rule]);
    setDraftColumns([]);
    setLookupTable("");
    setStrategy("lookup");
  }

  function handleRemove(index: number) {
    onSave(rules.filter((_, i) => i !== index));
  }

  return (
    <section aria-label={`correlations for ${table}`} className="db-subsection">
      <h4 className="db-subsection-title">Correlations</h4>
      <ul aria-label={`correlation rules for ${table}`} className="mb-3 flex flex-col gap-1">
        {rules.length === 0 && <li className="text-sm text-slate-500">No correlation rules.</li>}
        {rules.map((rule, i) => (
          <li
            key={`${rule.columns.join(",")}-${i}`}
            className="flex flex-wrap items-center gap-1 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm"
          >
            <span className="font-mono text-slate-700">{rule.columns.join(", ")}</span>
            {rule.lookup_table && <span className="text-slate-500"> → {rule.lookup_table}</span>}
            <span className="text-slate-400"> ({rule.strategy})</span>
            <button
              type="button"
              className="db-btn-danger ml-auto"
              aria-label={`remove correlation ${i + 1}`}
              onClick={() => handleRemove(i)}
            >
              remove
            </button>
          </li>
        ))}
      </ul>

      <fieldset className="db-fieldset flex flex-col gap-2">
        <legend className="db-legend">Add correlation</legend>
        <label className="db-field mb-0">
          <span className="db-label">columns</span>
          <select
            multiple
            aria-label="columns for new correlation"
            className="db-input"
            value={draftColumns}
            onChange={(e) =>
              setDraftColumns(Array.from(e.target.selectedOptions, (o) => o.value))
            }
          >
            {columns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        <label className="db-field mb-0">
          <span className="db-label">lookup table</span>
          <select
            aria-label="lookup table"
            className="db-input"
            value={lookupTable}
            onChange={(e) => setLookupTable(e.target.value)}
          >
            <option value="">(none)</option>
            {tables.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
        <label className="db-field mb-0">
          <span className="db-label">strategy</span>
          <input
            aria-label="correlation strategy"
            className="db-input"
            value={strategy}
            onChange={(e) => setStrategy(e.target.value)}
          />
        </label>
        {error && <p role="alert" className="db-notice-alert">{error}</p>}
        <button type="button" className="db-btn-secondary self-start" onClick={handleAdd}>
          add correlation
        </button>
      </fieldset>
    </section>
  );
}
