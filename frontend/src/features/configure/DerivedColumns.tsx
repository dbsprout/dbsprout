import { useState } from "react";
import type { DerivedColumn } from "../../api/types";

interface DerivedColumnsProps {
  /** Active table name (used for labelling + draft reset). */
  table: string;
  /** Column names available on the active table (for the depends_on picker). */
  columns: string[];
  /** Current derived columns for the active table. */
  derived: DerivedColumn[];
  /** Persist the full new derived list. */
  onSave: (derived: DerivedColumn[]) => void;
}

/**
 * Edits a table's expression-based derived columns. Keyed on `table` so the
 * add-derived draft resets cleanly when the user switches tables.
 */
export function DerivedColumns(props: DerivedColumnsProps) {
  return <Editor key={props.table} {...props} />;
}

function Editor({ table, columns, derived, onSave }: DerivedColumnsProps) {
  const [name, setName] = useState("");
  const [expression, setExpression] = useState("");
  const [dependsOn, setDependsOn] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  function handleAdd() {
    if (name.trim() === "" || expression.trim() === "") {
      setError("A derived column needs both a name and expression.");
      return;
    }
    setError(null);
    const col: DerivedColumn = {
      column: name.trim(),
      expression: expression.trim(),
      depends_on: dependsOn,
    };
    onSave([...derived, col]);
    setName("");
    setExpression("");
    setDependsOn([]);
  }

  function handleRemove(index: number) {
    onSave(derived.filter((_, i) => i !== index));
  }

  return (
    <section aria-label={`derived for ${table}`} className="db-subsection">
      <h4 className="db-subsection-title">Derived columns</h4>
      <ul aria-label={`derived columns for ${table}`} className="mb-3 flex flex-col gap-1">
        {derived.length === 0 && <li className="text-sm text-slate-500">No derived columns.</li>}
        {derived.map((col, i) => (
          <li
            key={`${col.column}-${i}`}
            className="flex flex-wrap items-center gap-1 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm"
          >
            <span className="font-mono text-slate-700">
              {col.column} = {col.expression}
            </span>
            {col.depends_on.length > 0 && (
              <span className="text-slate-400"> [{col.depends_on.join(", ")}]</span>
            )}
            <button
              type="button"
              className="db-btn-danger ml-auto"
              aria-label={`remove derived ${col.column}`}
              onClick={() => handleRemove(i)}
            >
              remove
            </button>
          </li>
        ))}
      </ul>

      <fieldset className="db-fieldset flex flex-col gap-2">
        <legend className="db-legend">Add derived column</legend>
        <label className="db-field mb-0">
          <span className="db-label">name</span>
          <input
            aria-label="derived column name"
            className="db-input"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <label className="db-field mb-0">
          <span className="db-label">expression</span>
          <input
            aria-label="derived expression"
            className="db-input font-mono"
            value={expression}
            onChange={(e) => setExpression(e.target.value)}
          />
        </label>
        <label className="db-field mb-0">
          <span className="db-label">depends on</span>
          <select
            multiple
            aria-label="depends on"
            className="db-input"
            value={dependsOn}
            onChange={(e) =>
              setDependsOn(Array.from(e.target.selectedOptions, (o) => o.value))
            }
          >
            {columns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        {error && <p role="alert" className="db-notice-alert">{error}</p>}
        <button type="button" className="db-btn-secondary self-start" onClick={handleAdd}>
          add derived
        </button>
      </fieldset>
    </section>
  );
}
