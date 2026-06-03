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
    <section aria-label={`correlations for ${table}`}>
      <h4>Correlations</h4>
      <ul aria-label={`correlation rules for ${table}`}>
        {rules.length === 0 && <li>No correlation rules.</li>}
        {rules.map((rule, i) => (
          <li key={`${rule.columns.join(",")}-${i}`}>
            <span>{rule.columns.join(", ")}</span>
            {rule.lookup_table && <span> → {rule.lookup_table}</span>}
            <span> ({rule.strategy})</span>
            <button
              type="button"
              aria-label={`remove correlation ${i + 1}`}
              onClick={() => handleRemove(i)}
            >
              remove
            </button>
          </li>
        ))}
      </ul>

      <fieldset>
        <legend>Add correlation</legend>
        <label>
          columns
          <select
            multiple
            aria-label="columns for new correlation"
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
        <label>
          lookup table
          <select
            aria-label="lookup table"
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
        <label>
          strategy
          <input
            aria-label="correlation strategy"
            value={strategy}
            onChange={(e) => setStrategy(e.target.value)}
          />
        </label>
        {error && <p role="alert">{error}</p>}
        <button type="button" onClick={handleAdd}>
          add correlation
        </button>
      </fieldset>
    </section>
  );
}
